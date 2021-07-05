import argparse
import logging
import os
import random
import time

import torch
import transformers
import numpy as np
from torchcrf import CRF
from tensorboardX import SummaryWriter
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# 设置随机种子，一旦固定种子，后面依次生成的随机数其实都是固定的
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# 定义数据处理的父类
class NERProcessor():

    def load_train_dataset(self, args, tokenizer, class_list):
        return None

    def load_dev_dataset(self, args, tokenizer, class_list):
        return None, None

    def load_test_dataset(self, args, tokenizer, class_list):
        return None, None

    def get_tag_list(self):
        return None

    # 加载train，dev，test文本文件（格式为：字 标签），返回的sentences是[[[字，标签]，[字，标签]，[字，标签]...]，[[]...],[[]...],[[]...]...]
    def _load_data(self, file_path, mode=None):
        sentences = []
        with open(file_path, "r", encoding="utf8") as fin:
            sentence = []
            for line in fin.readlines():
                data = line.replace('\n', '').split('\t')
                if len(data) == 2:
                    sentence.append(data)
                else:
                    sentences.append(sentence)
                    sentence = []
            if len(sentence) > 0:
                sentences.append(sentence)
        if mode:
            logging.info(mode + ": Read \"" + file_path + "\" sentence number:" + str(len(sentences)))
        else:
            logging.info("Read \"" + file_path + "\" sentence number:" + str(len(sentences)))
        return sentences

    # 这是span方法用到的函数
    # 每个句子都对应一个start_labels,长度为句长，记录的是每个实体开始的位置。参数data就是一个句子的labels。
    # 每个句子都对应一个end_labels，长度为句长，记录的是每个实体最后的位置。
    def _bio_to_se(self, data):
        start_labels = ['O'] * len(data)
        end_labels = ['O'] * len(data)
        i = 0
        while i < len(data):
            if data[i][0] == 'B':
                tag = data[i][2:]
                start_labels[i] = tag
                i += 1
                while i < len(data) and data[i] == "I-" + tag:
                    i += 1
                i -= 1
                end_labels[i] = tag
            i += 1
        return start_labels, end_labels

    # 对句子按照最大句长截断句子。如果是span架构，则返回result。如果是crf或者softmax架构，则返回new_data。
    # error_num表示的是超过最大句长但是没有标点的句子数目，这样的就强行截断。
    def _split(self, data, split_list, max_len, label_flag=False, mode="crf"):
        result = []
        new_data = []
        original_num = len(data)
        error_num = 0
        for sentence in data:
            data_list = []
            new_sentence = sentence
            while len(new_sentence) >= max_len:
                j = max_len - 1
                while j > 0:
                    if new_sentence[j][0] in split_list:
                        break
                    else:
                        j -= 1
                if j == 0:
                    j = max_len - 1
                    if label_flag:
                        while j + 1 >= len(new_sentence) or new_sentence[j + 1][1][0] == 'I':
                            j -= 1
                        if j == 0:
                            j = max_len - 1
                    error_num += 1
                data_list.append(new_sentence[:j + 1])
                new_sentence = new_sentence[j + 1:]
            if len(new_sentence) > 0:
                data_list.append(new_sentence)
            for data in data_list:
                text = []
                labels = []
                for char in data:
                    text.append(char[0])
                    labels.append(char[1])
                new_data.append((text, labels))
                if mode == "span":
                    start_labels, end_labels = self._bio_to_se(labels)
                    result.append((text, start_labels, end_labels))
        logging.info("final sentences: " + str(len(result)) + ", original sentences: " + str(original_num) +
                     ", error num:" + str(error_num))
        return result, new_data


# span架构所需的数据处理类
class SpanProcessor(NERProcessor):
    def load_train_dataset(self, args, tokenizer, class_list):
        data = self._load_data(args.train_file, "training")
        split_list = args.split.split()  # args参数定义那里有split的选项。所以split_list=[',', '，', '.', '。', '!', '！', '?', '？']
        data, _ = self._split(data, split_list, args.max_len - 2, True, mode="span")
        datasets = self._make_dataset(data, tokenizer, class_list, args.max_len, "Traning", 0)
        return datasets

    def load_dev_dataset(self, args, tokenizer, class_list):
        data = self._load_data(args.dev_file, "development")
        split_list = args.split.split()
        data, new_data = self._split(data, split_list, args.max_len - 2, True, mode="span")
        datasets = self._make_dataset(data, tokenizer, class_list, args.max_len, "development", 0)
        return datasets, new_data

    def load_test_dataset(self, args, tokenizer, class_list):
        original_data = self._load_data(args.test_file, "testing")
        split_list = args.split.split()
        data, _ = self._split(original_data, split_list, args.max_len - 2, False, mode="span")
        datasets = self._make_dataset(data, tokenizer, class_list, args.max_len, "Testing", 0)
        return datasets, original_data

    # span架构，取出data下的class.txt文件标签，并返回class_list。
    def get_tags_list(self, tags_file):
        class_list = []
        with open(tags_file, "r", encoding="utf8") as fin:
            for line in fin.readlines():
                data = line.replace("\n", "")
                if len(data) > 0:
                    class_list.append(data)
        logging.info("Class:" + str(class_list))
        return class_list

    # 把train，dev，test文本转化成ID号。返回的是token_ids_list，mask_ids_list，token_type_ids_list，start_labels_ids_list，end_labels_ids_list张量。
    def _make_dataset(self, data, tokenizer, class_list, max_len, mode, num=0):

        # 把句子中每个字的标签转换成标签列表中的ID号。
        def change_labels_to_ids(data, class_dict):
            result = [class_dict['O']]
            for label in data:
                result.append(class_dict[label])
            result.append(class_dict['O'])
            return result

        token_ids_list = []
        mask_ids_list = []
        token_type_ids_list = []
        start_labels_ids_list = []
        end_labels_ids_list = []
        class_dict = {x: i + 1 for i, x in enumerate(class_list)}
        class_dict['O'] = 0
        for i, sentence in enumerate(data):
            text = sentence[0]
            start_labels = sentence[1]
            end_labels = sentence[2]
            token_ids = tokenizer.encode(text)
            mask_ids = [1] * len(token_ids)
            start_label_ids = change_labels_to_ids(start_labels, class_dict)
            end_label_ids = change_labels_to_ids(end_labels, class_dict)
            assert len(start_label_ids) == len(end_label_ids) == len(token_ids)

            while len(token_ids) < max_len:
                token_ids.append(0)
                mask_ids.append(0)
                start_label_ids.append(class_dict['O'])
                end_label_ids.append(class_dict['O'])

            assert len(token_ids) == max_len
            assert len(mask_ids) == max_len
            assert len(start_label_ids) == max_len
            assert len(end_label_ids) == max_len

            if i < num:
                logging.info("*** " + mode + " Example - " + str(i + 1) + " - ***")
                logging.info("tokens: %s" % " ".join([str(x) for x in sentence[0]]))
                logging.info("token_ids: %s" % " ".join([str(x) for x in token_ids]))
                logging.info("mask_ids: %s" % " ".join([str(x) for x in mask_ids]))
                logging.info("token_type_ids_list: %s" % " ".join([str(x) for x in ([0] * len(token_ids))]))
                logging.info("start_label_ids: %s" % " ".join([str(x) for x in start_label_ids]))
                logging.info("end_label_ids: %s" % " ".join([str(x) for x in end_label_ids]))

            token_ids_list.append(token_ids)
            mask_ids_list.append(mask_ids)
            token_type_ids_list.append([0] * len(token_ids))
            start_labels_ids_list.append(start_label_ids)
            end_labels_ids_list.append(end_label_ids)

        return torch.utils.data.TensorDataset(
            torch.tensor(token_ids_list, dtype=torch.long),
            torch.tensor(mask_ids_list, dtype=torch.long),
            torch.tensor(token_type_ids_list, dtype=torch.long),
            torch.tensor(start_labels_ids_list, dtype=torch.long),
            torch.tensor(end_labels_ids_list, dtype=torch.long)
        )


# CRF架构所需的数据处理类
class CRFProcessor(NERProcessor):

    def load_train_dataset(self, args, tokenizer, tags_list):
        data = self._load_data(args.train_file, "training")
        split_list = args.split.split()
        _, data = self._split(data, split_list, args.max_len - 2, True, mode="crf")
        datasets = self._make_dataset(data, tokenizer, tags_list, args.max_len, "Traning", 5)
        return datasets

    def load_dev_dataset(self, args, tokenizer, class_list):
        data = self._load_data(args.dev_file, "development")
        split_list = args.split.split()
        _, data = self._split(data, split_list, args.max_len - 2, True, mode="crf")
        datasets = self._make_dataset(data, tokenizer, class_list, args.max_len, "development", 5)
        return datasets, data

    def load_test_dataset(self, args, tokenizer, class_list):
        original_data = self._load_data(args.test_file, "testing")
        split_list = args.split.split()
        _, data = self._split(original_data, split_list, args.max_len - 2, False, mode="crf")
        datasets = self._make_dataset(data, tokenizer, class_list, args.max_len, "Testing", 5)
        return datasets, original_data

    # CRF架构，取出data下tags标签，并加入<s>,<e>,<p>标签，返回tags_list
    def get_tags_list(self, tags_file):
        tags_list = []
        with open(tags_file, "r", encoding="utf8") as fin:
            for line in fin.readlines():
                data = line.replace("\n", "")
                if len(data) > 0:
                    tags_list.append(data)
        tags_list.append('<s>')
        tags_list.append('<e>')
        tags_list.append('<p>')
        logging.info("Tags:" + str(tags_list))
        return tags_list

    # 把train，dev，test文本转化成ID号。返回的是token_ids_list，mask_ids_list，token_type_ids_list，start_labels_ids_list，end_labels_ids_list张量。
    def _make_dataset(self, data, tokenizer, tags_list, max_len, mode, num=0):

        # 把句子中每个字的标签转换成标签列表中的ID号。
        def change_labels_to_ids(data, tags_dict):
            result = [tags_dict['<s>']]
            for label in data:
                result.append(tags_dict[label])
            result.append(tags_dict['<e>'])
            return result

        token_ids_list = []
        mask_ids_list = []
        token_type_ids_list = []
        labels_ids_list = []
        tags_dict = {x: i for i, x in enumerate(tags_list)}
        for i, sentence in enumerate(data):
            text = sentence[0]
            labels = sentence[1]
            token_ids = tokenizer.encode(text)
            mask_ids = [1] * len(token_ids)
            label_ids = change_labels_to_ids(labels, tags_dict)
            assert len(label_ids) == len(token_ids)

            while len(token_ids) < max_len:
                token_ids.append(0)
                mask_ids.append(0)
                label_ids.append(tags_dict['<p>'])

            assert len(token_ids) == max_len
            assert len(mask_ids) == max_len
            assert len(label_ids) == max_len

            if i < num:
                logging.info("*** " + mode + " Example - " + str(i + 1) + " - ***")
                logging.info("tokens: %s" % " ".join([str(x) for x in sentence[0]]))
                logging.info("token_ids: %s" % " ".join([str(x) for x in token_ids]))
                logging.info("mask_ids: %s" % " ".join([str(x) for x in mask_ids]))
                logging.info("token_type_ids_list: %s" % " ".join([str(x) for x in ([0] * len(token_ids))]))
                logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))

            token_ids_list.append(token_ids)
            mask_ids_list.append(mask_ids)
            token_type_ids_list.append([0] * len(token_ids))
            labels_ids_list.append(label_ids)

        return torch.utils.data.TensorDataset(
            torch.tensor(token_ids_list, dtype=torch.long),
            torch.tensor(mask_ids_list, dtype=torch.long),
            torch.tensor(token_type_ids_list, dtype=torch.long),
            torch.tensor(labels_ids_list, dtype=torch.long)
        )


# Softmax架构所需的数据处理类，可以继承CRF的数据处理类
class SoftMaxProcessor(CRFProcessor):

    # 取出softmax架构下tags标签，返回tags_list。这里与CRF稍有不同，不会加入<s>、<p>、<e>这三个多的标签。
    def get_tags_list(self, tags_file):
        tags_list = []
        with open(tags_file, "r", encoding="utf8") as fin:
            for line in fin.readlines():
                data = line.replace("\n", "")
                if len(data) > 0:
                    tags_list.append(data)
        logging.info("Tags:" + str(tags_list))
        return tags_list

    # 把train，dev，test文本转化成ID。返回的是token_ids_list，mask_ids_list，token_type_ids_list，start_labels_ids_list，end_labels_ids_list张量。
    def _make_dataset(self, data, tokenizer, tags_list, max_len, mode="softmax", num=0):

        # 把句子中每个字的标签转换成标签列表中的ID号。不用加开始、填充和结束字符。
        def change_labels_to_ids(data, tags_dict):
            result = [tags_dict['O']]
            for label in data:
                result.append(tags_dict[label])
            result.append(tags_dict['O'])
            return result

        token_ids_list = []
        mask_ids_list = []
        token_type_ids_list = []
        labels_ids_list = []
        tags_dict = {x: i for i, x in enumerate(tags_list)}
        for i, sentence in enumerate(data):
            text = sentence[0]
            labels = sentence[1]
            # 经过tokenizer.encode()会添加上特殊字符[CLS]和[SEP]，所以label_ids要补充两个O字符
            token_ids = tokenizer.encode(text)
            mask_ids = [1] * len(token_ids)
            label_ids = change_labels_to_ids(labels, tags_dict)
            assert len(label_ids) == len(token_ids)

            while len(token_ids) < max_len:
                token_ids.append(0)
                mask_ids.append(0)
                label_ids.append(tags_dict['O'])

            assert len(token_ids) == max_len
            assert len(mask_ids) == max_len
            assert len(label_ids) == max_len

            if i < num:
                logging.info("*** " + mode + " Example - " + str(i + 1) + " - ***")
                logging.info("tokens: %s" % " ".join([str(x) for x in sentence[0]]))
                logging.info("token_ids: %s" % " ".join([str(x) for x in token_ids]))
                logging.info("mask_ids: %s" % " ".join([str(x) for x in mask_ids]))
                logging.info("token_type_ids_list: %s" % " ".join([str(x) for x in ([0] * len(token_ids))]))
                logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))

            token_ids_list.append(token_ids)
            mask_ids_list.append(mask_ids)
            token_type_ids_list.append([0] * len(token_ids))
            labels_ids_list.append(label_ids)

        return torch.utils.data.TensorDataset(
            torch.tensor(token_ids_list, dtype=torch.long),
            torch.tensor(mask_ids_list, dtype=torch.long),
            torch.tensor(token_type_ids_list, dtype=torch.long),
            torch.tensor(labels_ids_list, dtype=torch.long)
        )


# NER模型的父类
class NERModel(torch.nn.Module):
    def __init__(self, bert_file_path):
        super(NERModel, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(bert_file_path)

    def forward(self, ids, mask, token_type_ids):
        return None

    def loss(self, output, label, mask_ids):
        return None


# span架构模型
class SpanModel(NERModel):
    def __init__(self, bert_file_path, config, class_num, dropout):
        super(SpanModel, self).__init__(bert_file_path)
        self.start_classifier = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(config.hidden_size, class_num),
            torch.nn.ReLU()
        )
        self.end_classifier = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(config.hidden_size, class_num),
            torch.nn.ReLU()
        )

    def forward(self, ids, mask, token_type_ids):
        output, _ = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        start_output = self.start_classifier(output)
        end_output = self.end_classifier(output)
        return start_output, end_output

    def loss(self, output, label, mask_ids):
        def calculate_loss(output, labels, mask_ids):
            bz, length = labels.shape
            mask = mask_ids.view(-1) == 1
            output = output.view(bz * length, -1)[mask]
            labels = labels.view(-1)[mask]
            return torch.nn.CrossEntropyLoss()(output, labels)
        start_output = output['start_output']
        start_labels_ids = label['start_labels_ids']
        end_output = output['end_output']
        end_labels_ids = label['end_labels_ids']
        loss_1 = calculate_loss(start_output, start_labels_ids, mask_ids)
        loss_2 = calculate_loss(end_output, end_labels_ids, mask_ids)
        return loss_1 + loss_2


# Softmax架构模型！注意在pytorch中，交叉熵函数，会自动加一层softmax激活函数
# 所以，训练时不需要自己把网络的输出结果再经过一次softmax了。解码的时候，也是选最大的，所以应该也不用softmax了。
class SoftmaxModel(NERModel):
    # 这个dropout和linear的顺序，应该影响不大
    def __init__(self, bert_file_path, config, tags_num, dropout):
        super(SoftmaxModel, self).__init__(bert_file_path)
        self.dence = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(config.hidden_size, tags_num),
            torch.nn.ReLU()
        )

    def forward(self, ids, mask, token_type_ids):
        output, _ = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        output = self.dence(output)
        return output

    def loss(self, output, labels_ids, mask_ids):
        bz, length = labels_ids.shape
        mask = mask_ids.view(-1) == 1
        output = output.view(bz * length, -1)[mask]
        labels = labels_ids.view(-1)[mask]
        loss = torch.nn.CrossEntropyLoss()(output, labels)
        return loss


# CRF架构模型
class CRFModel(NERModel):
    def __init__(self, bert_file_path, config, tags_num, dropout):
        super(CRFModel, self).__init__(bert_file_path)
        self.dence = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(config.hidden_size, tags_num),
            torch.nn.ReLU()
        )
        self.crf = CRF(tags_num, batch_first=True)

    def forward(self, ids, mask, token_type_ids):
        output, _ = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        output = self.dence(output)
        return output

    def loss(self, output, labels_ids, mask_ids):
        mask = mask_ids == 1
        return -self.crf(output, labels_ids, mask=mask, reduction='mean')

    def decode(self, output, mask_ids):
        mask = mask_ids == 1
        return self.crf.decode(output, mask)


# 如果有了模型，则进行验证，span，crf，softmax架构可选择
def development(model1, device, dev_dataloader, tags_list, mode):
    # 注意model.train()和model.eval()的不同作用。
    model1.eval()
    start_time = time.time()
    if mode == "span":
        start_labels_ids_list = []
        end_labels_ids_list = []
        start_output_list = []
        end_output_list = []
    elif mode == "softmax" or mode == "crf":
        labels_ids_list = []
        output_list = []
    mask_ids_list = []
    # tqdm进度条库，可视化
    for _, data in enumerate(tqdm(dev_dataloader, desc="Development")):
        token_ids = data[0].to(device, dtype=torch.long)
        mask_ids = data[1].to(device, dtype=torch.long)
        token_type_ids = data[2].to(device, dtype=torch.long)
        # print(token_ids.shape)
        # print(mask_ids.shape)
        # print(token_type_ids.shape)
        if mode == "span":
            start_labels_ids = data[3].to(device, dtype=torch.long)
            end_labels_ids = data[4].to(device, dtype=torch.long)
            with torch.no_grad():
                start_output, end_output = model1(token_ids, mask_ids, token_type_ids)
            start_output = start_output.argmax(dim=-1)
            end_output = end_output.argmax(dim=-1)

            start_labels_ids_list.append(start_labels_ids)
            end_labels_ids_list.append(end_labels_ids)
            start_output_list.append(start_output)
            end_output_list.append(end_output)
        elif mode == "softmax":
            labels_ids = data[3].to(device, dtype=torch.long)
            output = model1(token_ids, mask_ids, token_type_ids)
            output = output.argmax(dim=-1)

            output_list += output
            labels_ids_list.append(labels_ids)
        elif mode == "crf":
            labels_ids = data[3].to(device, dtype=torch.long)
            output = model1(token_ids, mask_ids, token_type_ids)
            output = model1.decode(output, mask_ids)

            labels_ids_list.append(labels_ids)
            output_list += output

        mask_ids_list.append(mask_ids)

    if mode == "span":
        start_labels_ids = torch.cat(start_labels_ids_list, dim=0)
        end_labels_ids = torch.cat(end_labels_ids_list, dim=0)
        start_outputs = torch.cat(start_output_list, dim=0)
        end_outputs = torch.cat(end_output_list, dim=0)
    elif mode == "crf" or mode == "softmax":
        labels_ids = torch.cat(labels_ids_list, dim=0)
    mask_ids_list = torch.cat(mask_ids_list, dim=0)

    outputs = {}
    labels = {}
    if mode == "span":
        outputs['start_outputs'] = start_outputs
        outputs['end_outputs'] = end_outputs
        outputs['num'] = start_labels_ids.size()[0]
        labels['start_labels_ids'] = start_labels_ids
        labels['end_labels_ids'] = end_labels_ids
    elif mode == "crf" or mode == "softmax":
        outputs['outputs'] = output_list
        outputs['num'] = len(output_list)
        labels['labels_ids'] = labels_ids
    f1, predict_list = evaluate(outputs, labels, mask_ids_list, tags_list, mode, False)

    end_time = time.time()
    logging.info("Development end, speed: {:.1f} sentences/s, all time: {:.2f}s".format(
        len(dev_dataloader) / (end_time - start_time), end_time - start_time))

    return f1, predict_list


# 训练过程
def train(args, model, device, train_datasets, dev_datasets, tags_list, writer):
    # 注意model.train()和model.eval()的不同作用。
    model.train()
    epoch_step = len(train_datasets) // args.train_batch_size + 1
    num_train_optimization_steps = epoch_step * args.epochs
    logging.info("***** Running training *****")
    logging.info("  Num examples = %d", len(train_datasets))
    logging.info("  Batch size = %d", args.train_batch_size)
    logging.info("  Num steps = %d", num_train_optimization_steps)

    # os.walk方法，主要用来遍历一个目录内各个子目录和子文件。
    # 可以得到一个三元tupple(dirpath, dirnames, filenames), 第一个为起始路径，第二个为起始路径下的文件夹，第三个是起始路径下的文件。
    _, _, files = list(os.walk(args.output_dir))[0]
    epoch = 0
    for file in files:
        if len(file) > 0 and file[:10] == "checkpoint":
            temp = file[11:-4]
            if temp.isdigit() and int(temp) > epoch:
                epoch = int(temp)
    # 如果训练了几轮，保存了模型，那就直接导入模型。
    if epoch > 0:
        logging.info('checkpoint-' + str(epoch) + '.pkl is exit!')
        model = torch.load(os.path.join(args.output_dir, 'checkpoint-' + str(epoch) + '.pkl'))
        logging.info("Load model:" + os.path.join(args.output_dir, 'checkpoint-' + str(epoch) + '.pkl'))

    if epoch >= args.epochs:
        logging.info("The model has been trained!")
        return

    train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=args.train_batch_size, shuffle=True)
    if args.dev_file:
        dev_dataloader = torch.utils.data.DataLoader(dev_datasets, batch_size=args.train_batch_size, shuffle=False)

    best_f1 = -1
    # 如果已经保存了最好的模型，就直接导入！
    if os.path.exists(os.path.join(args.output_dir, 'checkpoint-best.pkl')):
        logging.info('checkpoint-best.pkl is exit!')
        model = torch.load(os.path.join(args.output_dir, 'checkpoint-best.pkl'))
        best_f1, _ = development(model, device, dev_dataloader, tags_list, args.architecture)
        logging.info("Load best F1={:.4f}".format(best_f1))

    if args.architecture == "span":
        optimizer = transformers.AdamW(params=model.parameters(), lr=args.learning_rate)
    if args.architecture == "softmax":
        optimizer = transformers.AdamW(params=model.parameters(), lr=args.learning_rate)
    if args.architecture == "crf":
        optimizer = transformers.AdamW(
            params=[
                {'params': model.bert.parameters()},
                {'params': model.dence.parameters(), 'lr': args.crf_lr},
                {'params': model.crf.parameters(), 'lr': args.crf_lr}
            ],
            lr=args.learning_rate)
    # 学习率预热函数，使学习率线性增长，然后到某一schedule，在线性/指数降低
    lr_scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_train_optimization_steps) * args.warmup_proportion,
        num_training_steps=num_train_optimization_steps
    )

    # 开始训练
    for current_epoch in range(epoch, args.epochs):
        model.train()
        all_loss = 0
        start_time = time.time()
        for step, data in enumerate(tqdm(train_dataloader, desc="Training")):
            token_ids = data[0].to(device, dtype=torch.long)
            mask_ids = data[1].to(device, dtype=torch.long)
            token_type_ids = data[2].to(device, dtype=torch.long)
            if args.architecture == "span":
                start_labels_ids = data[3].to(device, dtype=torch.long)
                end_labels_ids = data[4].to(device, dtype=torch.long)
                label = {
                    "start_labels_ids": start_labels_ids,
                    "end_labels_ids": end_labels_ids
                }
                start_output, end_output = model(token_ids, mask_ids, token_type_ids)
                output = {
                    "start_output": start_output,
                    "end_output": end_output
                }
                loss = model.loss(output, label, mask_ids)
            elif args.architecture == "softmax":
                labels_ids = data[3].to(device, dtype=torch.long)
                output = model(token_ids, mask_ids, token_type_ids)
                loss = model.loss(output, labels_ids, mask_ids)
            elif args.architecture == "crf":
                labels_ids = data[3].to(device, dtype=torch.long)
                output = model(token_ids, mask_ids, token_type_ids)
                loss = model.loss(output, labels_ids, mask_ids)

            if writer:
                writer.add_scalar('loss', loss, global_step=current_epoch * epoch_step + step + 1)
                writer.add_scalar('learning_rate',
                                  optimizer.state_dict()['param_groups'][0]['lr'],
                                  global_step=current_epoch * epoch_step + step + 1)
            loss = loss/5
            all_loss += loss.item()
            loss.backward()
            if ((step + 1) % 5) == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()
            # optimizer.zero_grad()  # 把梯度置零，也就是把loss关于weight的导数变成0
            # optimizer的step为什么不能放在min-batch那个循环之外，还有optimizer.step和loss.backward的区别：
            # https://blog.csdn.net/xiaoxifei/article/details/87797935
            # loss.backward()
            # optimizer.step()
            # lr_scheduler.step()

        # pytorch 中的 state_dict 是一个简单的python的字典对象,将每一层与它的对应参数建立映射关系。
        # torch.optim模块中的Optimizer优化器对象也存在一个state_dict对象，此处的state_dict字典对象包含state和param_groups的字典对象，
        # 而param_groups key对应的value也是一个由学习率，动量等参数组成的一个字典对象。
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        end_time = time.time()
        logging.info("Epoch: {}, Loss: {:.3g}, learning rate: {:.3g}, Time: {:.2f}s".format(
            current_epoch + 1, all_loss / (step + 1), lr, end_time - start_time))
        torch.save(model, os.path.join(args.output_dir, 'checkpoint-' + str(current_epoch + 1) + '.pkl'))
        delet_checkpoints_name = os.path.join(args.output_dir, 'checkpoint-' + str(
            current_epoch + 1 - args.keep_last_n_checkpoints) + '.pkl')
        if os.path.exists(delet_checkpoints_name):
            os.remove(delet_checkpoints_name)
        if args.dev_file:
            f1, _ = development(model, device, dev_dataloader, tags_list, args.architecture)
            if f1 == -1 or f1 > best_f1:
                best_f1 = f1
                logging.info("Best F1={:.4f}, save model!".format(best_f1))
                torch.save(model, os.path.join(args.output_dir, 'checkpoint-best.pkl'))
            if writer:
                writer.add_scalar('dev_f1', f1, global_step=current_epoch * epoch_step)
                writer.add_scalar('dev_best_f1', best_f1, global_step=current_epoch * epoch_step)

    torch.save(model, os.path.join(args.output_dir, 'checkpoint-last.pkl'))
    if args.dev_file:
        f1, _ = development(model, device, dev_dataloader, tags_list, args.architecture)
        if f1 == -1 or f1 > best_f1:
            best_f1 = f1
            logging.info("Best F1={:.4f}, save model!".format(best_f1))
            torch.save(model, os.path.join(args.output_dir, 'checkpoint-best.pkl'))
    logging.info("Training end!")


# 计算评价指标：准确率，召回率，F1值
def calculate(data):
    p = -1
    r = -1
    f1 = -1
    if data[0] > 0:
        p = data[2] / data[0]
    if data[1] > 0:
        r = data[2] / data[1]
    if p != -1 and r != -1 and p + r != 0:
        f1 = 2 * p * r / (p + r)
    return p, r, f1


# 评价过程，打印各类实体F1值和总体F1值
def evaluate(outputs, labels, mask_ids, tags_list, mode, test_flag=False):
    def change_label_span(start_tags, end_tags, length):
        i = 0
        result = []
        while i < length:
            if start_tags[i] != 0:
                tag = start_tags[i]
                start_index = i
                while i < length and end_tags[i] == 0:
                    i += 1
                if i < length and end_tags[i] == tag:
                    result.append((start_index, i + 1, tag.item()))
            i += 1
        return result

    # CRF架构和softmax架构共用
    def change_label_bio(data, length, tags_list):
        i = 0
        result = []
        while i < length:
            if tags_list[data[i]][0] == 'B':
                tag = tags_list[data[i]][2:]
                start_index = i
                i += 1
                while i < length and tags_list[data[i]] == 'I-' + tag:
                    i += 1
                result.append((start_index, i, tag))
                i -= 1
            i += 1
        return result

    result_f1 = None
    sentence_num = outputs['num']
    if mode == 'span':
        entities_dict = {x + 1: [0, 0, 0] for x in range(len(tags_list))}
    elif mode == 'crf' or mode == "softmax":
        entities_dict = {}
        for tag in tags_list:
            if tag[0] == 'B':
                entities_dict[tag[2:]] = [0, 0, 0]
    result_list = []
    for i in range(sentence_num):
        if mode == 'span':
            length = mask_ids[i].sum()
            predict_list = change_label_span(outputs['start_outputs'][i], outputs['end_outputs'][i], length)
        elif mode == 'softmax':
            # 注意解码长度的问题。crf是人家写好的，正好解码句长+2个标签。所以softmax方法应该和span方法一样，取length=mask_ids[i].sum()。不然会按最大句长解码。
            length = mask_ids[i].sum()
            predict_list = change_label_bio(outputs['outputs'][i], length, tags_list)
        elif mode == 'crf':
            length = len(outputs['outputs'][i])
            predict_list = change_label_bio(outputs['outputs'][i], length, tags_list)
        result_list.append((predict_list, length - 2))
        if not test_flag:
            if mode == 'span':
                label_list = change_label_span(labels['start_labels_ids'][i], labels['end_labels_ids'][i], length)
            elif mode == 'softmax' or mode == 'crf':
                label_list = change_label_bio(labels['labels_ids'][i], length, tags_list)
            for label in label_list:
                entities_dict[label[2]][1] += 1
            for predict in predict_list:
                entities_dict[predict[2]][0] += 1
                if predict in label_list:
                    entities_dict[predict[2]][2] += 1
    if not test_flag:
        all_result = [0, 0, 0]
        for entity in entities_dict:
            for i in range(len(entities_dict[entity])):
                all_result[i] += entities_dict[entity][i]
        logging.info("***** Development Evaluation *****")
        p, r, f1 = calculate(all_result)
        logging.info("ALL Precision={:.4f}, Recall={:.4f}, F1={:.4f}, predict: {}, truth: {}, right: {}".format(
            p, r, f1, all_result[0], all_result[1], all_result[2]))
        result_f1 = f1
        for tag_type in entities_dict:
            if mode == "span":
                tag = tags_list[tag_type - 1]
            elif mode == 'softmax':
                tag = tag_type
            elif mode == "crf":
                tag = tag_type
            p, r, f1 = calculate(entities_dict[tag_type])
            logging.info("{} Precision={:.4f}, Recall={:.4f}, F1={:.4f}, predict: {}, truth: {}, "
                         "right: {}".format(tag, p, r, f1, entities_dict[tag_type][0],
                                            entities_dict[tag_type][1], entities_dict[tag_type][2]))
    return result_f1, result_list


# 验证过程，直接加载模型
def dev(args, datasets, model, device, tags_list, sentences):
    # 加载模型，没有模型则报错
    if args.model is not None:
        model = torch.load(args.model)
        logging.info("Load model:" + args.model)
    elif os.path.exists(os.path.join(args.output_dir, 'checkpoint-best.pkl')):
        model = torch.load(os.path.join(args.output_dir, 'checkpoint-best.pkl'))
        logging.info("Load model:" + os.path.join(args.output_dir, 'checkpoint-best.pkl'))
    elif os.path.exists(os.path.join(args.output_dir, 'checkpoint-last.pkl')):
        model = torch.load(os.path.join(args.output_dir, 'checkpoint-last.pkl'))
        logging.info("Load model:" + os.path.join(args.output_dir, 'checkpoint-last.pkl'))
    else:
        logging.info("Error! The model file does not exist!")
        exit(1)
    model.eval()
    dataloader = torch.utils.data.DataLoader(datasets, batch_size=args.test_batch_size, shuffle=False)
    _, predict_list = development(model, device, dataloader, tags_list, args.architecture)
    assert len(sentences) == len(predict_list)
    write_data_list = []
    for i in range(len(sentences)):
        sentence, label = sentences[i]
        predict, _ = predict_list[i]
        result = ['O'] * len(sentence)
        for entity in predict:
            if args.architecture == 'span':
                tag = tags_list[entity[2] - 1]
            elif args.architecture == 'softmax':
                tag = entity[2]
            elif args.architecture == 'crf':
                tag = entity[2]
            result[entity[0] - 1] = "B-" + tag
            for j in range(entity[0], entity[1] - 1):
                result[j] = "I-" + tag
        write_data = []
        for j in range(len(sentence)):
            write_data.append((sentence[j], label[j], result[j]))
        write_data_list.append(write_data)

    with open(os.path.join(args.output_dir, "development.txt"), "w", encoding="utf8") as fout:
        for sentence in write_data_list:
            for data in sentence:
                fout.write(data[0] + '\t' + data[1] + '\t' + data[2] + '\n')
            fout.write('\n')
    logging.info("Development data is written to file: " + os.path.join(args.output_dir, "development.txt") + '!')


# 测试过程，直接加载模型
def test(args, processor, tokenizer, model, device, tags_list):
    if args.model is not None:
        model = torch.load(args.model)
        logging.info("Load model:" + args.model)
    elif os.path.exists(os.path.join(args.output_dir, 'checkpoint-best.pkl')):
        model = torch.load(os.path.join(args.output_dir, 'checkpoint-best.pkl'))
        logging.info("Load model:" + os.path.join(args.output_dir, 'checkpoint-best.pkl'))
    elif os.path.exists(os.path.join(args.output_dir, 'checkpoint-last.pkl')):
        model = torch.load(os.path.join(args.output_dir, 'checkpoint-last.pkl'))
        logging.info("Load model:" + os.path.join(args.output_dir, 'checkpoint-last.pkl'))
    else:
        logging.info("Error! The model file does not exist!")
        exit(1)

    start_time = time.time()
    datasets, sentences = processor.load_test_dataset(args, tokenizer, tags_list)
    model.eval()
    logging.info("***** Running Testing *****")
    logging.info("  Num examples = %d", len(datasets))
    logging.info("  Batch size = %d", args.test_batch_size)

    dataloader = torch.utils.data.DataLoader(datasets, batch_size=args.test_batch_size, shuffle=False)
    if args.architecture == "span":
        start_labels_ids_list = []
        end_labels_ids_list = []
        start_output_list = []
        end_output_list = []
    elif args.architecture == 'softmax':
        labels_ids_list = []
        output_list = []
    elif args.architecture == "crf":
        labels_ids_list = []
        output_list = []
    mask_ids_list = []
    for _, data in enumerate(tqdm(dataloader, desc="Testing")):
        token_ids = data[0].to(device, dtype=torch.long)
        mask_ids = data[1].to(device, dtype=torch.long)
        token_type_ids = data[2].to(device, dtype=torch.long)
        if args.architecture == "span":
            start_labels_ids = data[3].to(device, dtype=torch.long)
            end_labels_ids = data[4].to(device, dtype=torch.long)
        elif args.architecture == 'softmax':
            labels_ids = data[3].to(device, dtype=torch.long)
        elif args.architecture == "crf":
            labels_ids = data[3].to(device, dtype=torch.long)
        mask_ids_list.append(mask_ids)

        if args.architecture == "span":
            with torch.no_grad():
                start_output, end_output = model(token_ids, mask_ids, token_type_ids)
            start_output = start_output.argmax(dim=-1)
            end_output = end_output.argmax(dim=-1)

            start_labels_ids_list.append(start_labels_ids)
            end_labels_ids_list.append(end_labels_ids)
            start_output_list.append(start_output)
            end_output_list.append(end_output)
        elif args.architecture == 'softmax':
            labels_ids_list.append(labels_ids)
            output = model(token_ids, mask_ids, token_type_ids)
            output = output.argmax(dim=-1)
            output_list += output
        elif args.architecture == "crf":
            labels_ids_list.append(labels_ids)
            output = model(token_ids, mask_ids, token_type_ids)
            output = model.decode(output, mask_ids)
            output_list += output

    if args.architecture == "span":
        start_labels_ids = torch.cat(start_labels_ids_list, dim=0)
        end_labels_ids = torch.cat(end_labels_ids_list, dim=0)
        start_outputs = torch.cat(start_output_list, dim=0)
        end_outputs = torch.cat(end_output_list, dim=0)
    elif args.architecture == 'softmax':
        labels_ids = torch.cat(labels_ids_list, dim=0)
    elif args.architecture == "crf":
        labels_ids = torch.cat(labels_ids_list, dim=0)
    mask_ids_list = torch.cat(mask_ids_list, dim=0)

    outputs = {}
    labels = {}
    if args.architecture == "span":
        outputs['start_outputs'] = start_outputs
        outputs['end_outputs'] = end_outputs
        outputs['num'] = start_labels_ids.size()[0]
        labels['start_labels_ids'] = start_labels_ids
        labels['end_labels_ids'] = end_labels_ids
    elif args.architecture == 'softmax':
        outputs['outputs'] = output_list
        outputs['num'] = len(output_list)
        labels['labels_ids'] = labels_ids
    elif args.architecture == "crf":
        outputs['outputs'] = output_list
        outputs['num'] = len(output_list)
        labels['labels_ids'] = labels_ids
    _, predict_list = evaluate(outputs, labels, mask_ids_list, tags_list, args.architecture, True)

    write_data_list = []
    m = 0
    for i in range(len(sentences)):
        sentence = sentences[i]
        result = []
        while len(result) < len(sentence):
            predict, length = predict_list[m]
            m += 1
            predict_tags = ['O'] * length
            for entity in predict:
                if args.architecture == 'span':
                    tag = tags_list[entity[2] - 1]
                elif args.architecture == 'softmax':
                    tag = entity[2]
                elif args.architecture == 'crf':
                    tag = entity[2]
                predict_tags[entity[0] - 1] = "B-" + tag
                for j in range(entity[0], entity[1] - 1):
                    predict_tags[j] = "I-" + tag
            result += predict_tags
        assert len(result) == len(sentence)
        write_data = []
        for j in range(len(sentence)):
            write_data.append((sentence[j][0], sentence[j][1], result[j]))
        write_data_list.append(write_data)

    with open(os.path.join(args.output_dir, "test2.txt"), "w", encoding="utf8") as fout:
        for sentence in write_data_list:
            for data in sentence:
                fout.write(data[0] + '\t' + data[1] + '\t' + data[2] + '\n')
            fout.write('\n')
    logging.info("Prediction data is written to file: " + os.path.join(args.output_dir, "test2.txt") + '!')

    end_time = time.time()
    logging.info("Testing end, speed: {:.1f} sentences/s, all time: {:.2f}s".format(
        len(datasets) / (end_time - start_time), end_time - start_time))

    def change_label(tags):
        i = 0
        result = []
        while i < len(tags):
            if tags[i][0] == "B":
                class_type = tags[i][2:]
                start_index = i
                i += 1
                while i < len(tags) and tags[i] == "I-" + class_type:
                    i += 1
                result.append((start_index, i, class_type))
                i -= 1
            i += 1
        return result

    if args.architecture == "span":
        entities_dict = {x: [0, 0, 0] for x in tags_list}
    else:
        entities_dict = {}
        for tag in tags_list:
            if tag[0] == 'B':
                entities_dict[tag[2:]] = [0, 0, 0]
    for sentence in write_data_list:
        predict_data = []
        label_data = []
        for data in sentence:
            predict_data.append(data[2])
            label_data.append(data[1])
        predict_list = change_label(predict_data)
        label_list = change_label(label_data)
        for label in label_list:
            entities_dict[label[2]][1] += 1
        for predict in predict_list:
            entities_dict[predict[2]][0] += 1
            if predict in label_list:
                entities_dict[predict[2]][2] += 1
    all_result = [0, 0, 0]
    for entity in entities_dict:
        for i in range(len(entities_dict[entity])):
            all_result[i] += entities_dict[entity][i]
    logging.info("***** Testing Evaluation *****")
    p, r, f1 = calculate(all_result)
    logging.info("ALL Precision={:.4f}, Recall={:.4f}, F1={:.4f}, predict: {}, truth: {}, right: {}".format(
        p, r, f1, all_result[0], all_result[1], all_result[2]))
    for tag_type in entities_dict:
        p, r, f1 = calculate(entities_dict[tag_type])
        logging.info("{} Precision={:.4f}, Recall={:.4f}, F1={:.4f}, predict: {}, truth: {}, "
                     "right: {}".format(tag_type, p, r, f1, entities_dict[tag_type][0],
                                        entities_dict[tag_type][1], entities_dict[tag_type][2]))


def main():
    parser = argparse.ArgumentParser(description="Named Entity Recognition")

    parser.add_argument("--train_file", default=None, help="The training file path.")
    parser.add_argument("--dev_file", default=None, help="The development file path.")
    parser.add_argument("--test_file", default=None, help="The testing file path.")
    parser.add_argument("--tags_file", required=True, help="The tags file path.")

    parser.add_argument("--output_dir", required=True, help="The output folder path.")

    parser.add_argument("--model", default=None, help="The model path.")

    parser.add_argument("--architecture", default="span", choices=['span', 'crf', 'softmax'],
                        help="The model architecture of neural network and what decoding method is adopted.")
    parser.add_argument("--train_batch_size", default=1, type=int,
                        help="The number of sentences contained in a batch during training.")
    parser.add_argument("--test_batch_size", default=1, type=int,
                        help="The number of sentences contained in a batch during testing.")
    parser.add_argument("--epochs", default=1, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--crf_lr", default=0.0001, type=float,
                        help="The initial learning rate of CRF layer.")
    parser.add_argument("--max_len", required=True, type=int, help="The Maximum length of a sentence.")
    parser.add_argument("--dropout", default=0.0, type=float,
                        help="What percentage of neurons are discarded in the fully connected layers (0 ~ 1).")
    parser.add_argument("--keep_last_n_checkpoints", default=1, type=int,
                        help="Keep the last n checkpoints.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. ")
    parser.add_argument("--split", default="", help="Characters that segments a sentence.")
    parser.add_argument("--tensorboard_dir", default=None, help="The data address of the tensorboard.")

    parser.add_argument("--bert_config_file", required=True,
                        help="The config json file corresponding to the pre-trained BERT model. "
                             "This specifies the model architecture.")
    parser.add_argument("--cpu", default=False, action='store_true',
                        help="Whether to use CPU, if not and CUDA is avaliable can use CPU.")
    parser.add_argument('--seed', type=int, default=0,
                        help="random seed for initialization.")

    args = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(funcName)s: %(message)s',
        datefmt='%m-%d-%Y-%H:%M:%S',
        filemode='w',
        level=logging.INFO
    )

    setting = vars(args)
    logging.info("-" * 20 + "args" + "-" * 20)
    for key, value in setting.items():
        logging.info('%-30s%-s' % (key, str(value)))

    # Set seed
    set_seed(args.seed)

    if not args.train_file and not args.dev_file and not args.test_file:
        raise ValueError("At least one of `train_file`, `dev_file` or `test_file` must be not None.")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    logging.info("Output directory: " + args.output_dir)

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    logging.info("device: " + str(device))

    architecture = args.architecture

    tokenizer = transformers.BertTokenizer.from_pretrained(args.bert_config_file)
    config = transformers.BertConfig(args.bert_config_file)

    processor = None
    tags_list = None
    model = None
    if architecture == "span":
        processor = SpanProcessor()
        tags_list = processor.get_tags_list(args.tags_file)
        model = SpanModel(args.bert_config_file, config, len(tags_list) + 1, args.dropout)
    elif architecture == 'softmax':
        processor = SoftMaxProcessor()
        tags_list = processor.get_tags_list(args.tags_file)
        model = SoftmaxModel(args.bert_config_file, config, len(tags_list), args.dropout)
    elif architecture == "crf":
        processor = CRFProcessor()
        tags_list = processor.get_tags_list(args.tags_file)
        model = CRFModel(args.bert_config_file, config, len(tags_list), args.dropout)

    model.to(device)
    logging.info(model)

    writer = None
    if args.tensorboard_dir:
        writer = SummaryWriter(args.tensorboard_dir)
        # writer.add_graph(model, (torch.zeros(1, 10).to(device).long(),
        #                          torch.zeros(1, 10).to(device).long(),
        #                          torch.zeros(1, 10).to(device).long()))

    train_datasets = None
    dev_datasets, dev_data = None, None
    # 只使用 --train_file 则只训练到固定轮数，保存为最后的模型 checkpoint-last.kpl
    # if args.train_file:
    #     train_datasets = processor.load_train_dataset(args, tokenizer, tags_list)
    # 若使用 --train_file 和 --dev_file 则会额外域保存在开发集上的最高分数的模型 checkpoint-best.kpl
    # if args.dev_file:
    #     dev_datasets, dev_data = processor.load_dev_dataset(args, tokenizer, tags_list)
    # if args.train_file:
    #     train(args, model, device, train_datasets, dev_datasets, tags_list, writer)
    # if args.dev_file:
    #     dev(args, dev_datasets, model, device, tags_list, dev_data)
    if args.test_file:
        print("run test file")
        test(args, processor, tokenizer, model, device, tags_list)


if __name__ == "__main__":
    main()
