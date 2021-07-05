# Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
classification
"""
import ast
import os
import csv
import json
import warnings
import random
import argparse
import traceback
from functools import partial
from collections import namedtuple
from metric_sequence import Accuracy
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_transformers import BertTokenizer,BertConfig
from pytorch_transformers import AdamW, WarmupLinearSchedule
from torch.utils.data import Dataset, DataLoader, RandomSampler
from model import BertForSequenceClassification
from utils import read_by_lines, write_by_lines, load_dict

# warnings.filterwarnings('ignore')
"""
For All pre-trained model（English and Chinese),
Please refer to https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/transformers.md.
"""

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--num_epoch", type=int, default=3, help="Number of epoches for fine-tuning.")
parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate used to train with warmup.")
parser.add_argument("--tag_path", type=str, default='./conf/DuEE-Fin/enum_tag.dict', help="tag set path")
parser.add_argument("--train_data", type=str, default='./data/DuEE-Fin/enum/train.tsv', help="train data")
parser.add_argument("--dev_data", type=str, default='./data/DuEE-Fin/enum/dev.tsv', help="dev data")
parser.add_argument("--test_data", type=str, default='./data/DuEE-Fin/enum/test1.tsv', help="test data")
parser.add_argument("--predict_data", type=str, default=None, help="predict data")
parser.add_argument("--do_train", type=ast.literal_eval, default=False, help="do train")
parser.add_argument("--do_predict", type=ast.literal_eval, default=True, help="do predict")
parser.add_argument("--pretrain_model", type=str, default='/home/xuwd/data/bert-base-chinese/', help="pretrain language model name or path")

parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay rate for L2 regularizer.")
parser.add_argument("--warmup_proportion", type=float, default=0.1, help="Warmup proportion params for warmup strategy")
parser.add_argument("--max_seq_len", type=int, default=512, help="Number of words of the longest seqence.")
parser.add_argument("--valid_step", type=int, default=100, help="validation step")
parser.add_argument("--skip_step", type=int, default=20, help="skip step")
parser.add_argument("--batch_size", type=int, default=32, help="Total examples' number in batch for training.")
parser.add_argument("--checkpoints", type=str, default='./ckpt/DuEE-Fin/enum/', help="Directory to model checkpoint")
parser.add_argument("--init_ckpt", type=str, default='./ckpt/DuEE-Fin/enum/best.pdparams', help="already pretraining model checkpoint")
parser.add_argument("--predict_save_path", type=str, default='./ckpt/DuEE-Fin/enum/test1_pred.json', help="predict data save path")
parser.add_argument("--seed", type=int, default=1000, help="random seed for initialization")
parser.add_argument("--n_gpu", type=int, default=1, help="Number of GPUs to use, 0 for CPU.")
args = parser.parse_args()
# yapf: enable.

def set_seed(random_seed):
    """sets random seed"""
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


@torch.no_grad()
def evaluate(model, criterion, metric, data_loader):
    """
    Given a dataset, it evals model and computes the metric.
    Args:
        model(obj:`paddle.nn.Layer`): A model to classify texts.
        data_loader(obj:`paddle.io.DataLoader`): The dataset loader which generates batches.
        criterion(obj:`paddle.nn.Layer`): It can compute the loss.
        metric(obj:`paddle.metric.Metric`): The evaluation metric.
    """
    model.eval()
    metric.reset()
    losses = []
    for batch in data_loader:
        input_ids, segment_ids, mask_ids, labels = batch
        logits = model(input_ids, segment_ids, mask_ids)
        loss = criterion(logits, labels)
        losses.append(loss.item())
        metric.compute(logits, labels)

    accuracy = metric.accumulate()
    metric.reset()
    model.train()
    return float(np.mean(losses)), accuracy


def convert_example(example, tokenizer, label_map=None, max_seq_len=512, is_test=False):
    """convert_example"""

    batch_input_ids, batch_segment_ids, batch_mask_ids = [], [], []
    batch_labels = []

    for i in range(len(example)):
        tokens = example[i][0]

        tokens = tokens[:max_seq_len - 2]
        tokens = ['[CLS]'] + tokens + ['[SEP]']

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        length = len(input_ids)
        segment_ids = [0] * length
        mask_ids = [1] * length

        batch_input_ids.append(input_ids)
        batch_segment_ids.append(segment_ids)
        batch_mask_ids.append(mask_ids)
    batch_lengths = [len(x) for x in batch_input_ids]
    max_len = max(batch_lengths)
    batch_input_ids = torch.LongTensor([ids + [0] * (max_len - len(ids)) for ids in batch_input_ids]).cuda()
    batch_segment_ids = torch.LongTensor([ids + [0] * (max_len - len(ids)) for ids in batch_segment_ids]).cuda()
    batch_mask_ids = torch.LongTensor([ids + [0] * (max_len - len(ids)) for ids in batch_mask_ids]).cuda()


    if is_test:
        return batch_input_ids, batch_segment_ids, batch_mask_ids
    else:
        for i in range(len(example)):
            labels = label_map[example[i][1]]

            batch_labels.append(labels)

        batch_labels = torch.LongTensor(batch_labels).cuda()
        return batch_input_ids, batch_segment_ids, batch_mask_ids, batch_labels


class DuEventExtraction(Dataset):
    """Du"""

    def __init__(self, data_path, tag_path):
        self.label_vocab = load_dict(tag_path)
        self.word_ids = []
        self.label_ids = []
        with open(data_path, 'r', encoding='utf-8') as fp:
            # skip the head line
            next(fp)
            for line in fp.readlines():
                label, words = line.strip('\n').split('\t')
                words = words.split('\002')

                self.word_ids.append(words)
                self.label_ids.append(label)
        self.label_num = max(self.label_vocab.values()) + 1

    def __len__(self):
        return len(self.word_ids)

    def __getitem__(self, index):
        return self.word_ids[index], self.label_ids[index]


'''def data_2_examples(datas):
    """data_2_examples"""
    has_text_b, examples = False, []
    if isinstance(datas[0], list):
        Example = namedtuple('Example', ["text_a", "text_b"])
        has_text_b = True
    else:
        Example = namedtuple('Example', ["text_a"])
    for item in datas:
        if has_text_b:
            example = Example(text_a=item[0], text_b=item[1])
        else:
            example = Example(text_a=item)
        examples.append(example)
    return examples'''


def do_train():
    set_seed(args.seed)
    torch.device("cuda" if args.n_gpu  else "cpu")


    label_map = load_dict(args.tag_path)
    id2label = {val: key for key, val in label_map.items()}

    tokenizer = BertTokenizer.from_pretrained(args.pretrain_model)
    config = BertConfig.from_pretrained(args.pretrain_model, num_labels=len(label_map))
    model = BertForSequenceClassification.from_pretrained(args.pretrain_model, config=config)
    model.cuda()
    model.zero_grad()

    print("============start train==========")
    train_ds = DuEventExtraction(args.train_data, args.tag_path)
    dev_ds = DuEventExtraction(args.dev_data, args.tag_path)
    test_ds = DuEventExtraction(args.test_data, args.tag_path)

    trans_func = partial(
        convert_example, tokenizer=tokenizer, label_map=label_map, max_seq_len=args.max_seq_len)
    #batchify_fn = lambda samples: trans_func(samples)

    batch_sampler = RandomSampler(train_ds)
    train_loader = DataLoader(
        dataset=train_ds,
        sampler=batch_sampler,
        batch_size=args.batch_size,
        collate_fn=trans_func)
    dev_loader = DataLoader(
        dataset=dev_ds,
        batch_size=args.batch_size,
        collate_fn=trans_func)
    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=args.batch_size,
        collate_fn=trans_func)

    num_training_steps = len(train_loader) * args.num_epoch
    metric = Accuracy()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(),lr=args.learning_rate)

    step, best_performerence = 0, 0.0
    model.train()
    for epoch in range(args.num_epoch):
        for idx, (input_ids, segment_ids, mask_ids, labels) in enumerate(train_loader):
            logits = model(input_ids, segment_ids, mask_ids)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()
            model.zero_grad()
            loss_item = loss.item()
            if step > 0 and step % args.skip_step == 0:
                print(f'train epoch: {epoch} - step: {step} (total: {num_training_steps}) ' \
                    f'- loss: {loss_item:.6f}')
            if step > 0 and step % args.valid_step == 0 :
                loss_dev, acc_dev = evaluate(model, criterion, metric, dev_loader)
                print(f'dev step: {step} - loss: {loss_dev:.6f} accuracy: {acc_dev:.5f}, ' \
                        f'current best {best_performerence:.5f}')
                if acc_dev > best_performerence:
                    best_performerence = acc_dev
                    print(f'==============================================save best model ' \
                            f'best performerence {best_performerence:5f}')
                    torch.save(model.state_dict(), '{}/best.pdparams'.format(args.checkpoints))
            step += 1

    # save the final model

    torch.save(model.state_dict(), '{}/final.pdparams'.format(args.checkpoints))


def do_predict():
    set_seed(args.seed)
    torch.device("cuda" if args.n_gpu  else "cpu")

    label_map = load_dict(args.tag_path)
    id2label = {val: key for key, val in label_map.items()}

    tokenizer = BertTokenizer.from_pretrained(args.pretrain_model)
    config = BertConfig.from_pretrained(args.pretrain_model, num_labels=len(label_map))
    model = BertForSequenceClassification.from_pretrained(args.pretrain_model, config=config)
    model.cuda()


    print("============start predict==========")
    if not args.init_ckpt or not os.path.isfile(args.init_ckpt):
        raise Exception("init checkpoints {} not exist".format(args.init_ckpt))
    else:
        model.load_state_dict(torch.load(args.init_ckpt))
        print("Loaded parameters from %s" % args.init_ckpt)

    # load data from predict file
    sentences = read_by_lines(args.predict_data) # origin data format
    sentences = [json.loads(sent) for sent in sentences]
    examples = []

    for sent in sentences:
        text = sent["text"].split(" ")
        text = ','.join(text)
        text = [
            "，" if t == " " or t == "\n" or t == "\t" else t
            for t in list(text)
        ]
        sent["text"] = "".join(text)

        #input_sent = [text]  # only text_a

        #example = data_2_examples(input_sent)[0]
        examples.append((text,[]))

    # Seperates data into some batches.
    batch_encoded_inputs = [examples[i: i + args.batch_size]
                            for i in range(0, len(examples), args.batch_size)]
    results = []
    model.eval()
    for batch in batch_encoded_inputs:
        input_ids, segment_ids, mask_ids = convert_example(batch, tokenizer,
                    max_seq_len=args.max_seq_len, is_test=True)
        with torch.no_grad():
            logits = model(input_ids, segment_ids, mask_ids)
        probs = F.softmax(logits, dim=1)
        probs_ids = torch.argmax(probs, -1).cpu().numpy()
        probs = probs.detach().cpu().numpy()
        for prob_one, p_id in zip(probs.tolist(), probs_ids.tolist()):
            label_probs = {}
            for idx, p in enumerate(prob_one):
                label_probs[id2label[idx]] = p
            results.append({"probs": label_probs, "label": id2label[p_id]})

    assert len(results) == len(sentences)
    for sent, ret in zip(sentences, results):
        sent["pred"] = ret
    sentences = [json.dumps(sent, ensure_ascii=False) for sent in sentences]
    write_by_lines(args.predict_save_path, sentences)
    print("save data {} to {}".format(len(sentences), args.predict_save_path))


if __name__ == '__main__':

    if args.do_train:
        do_train()
    elif args.do_predict:
        do_predict()
