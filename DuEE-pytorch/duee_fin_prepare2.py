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
"""duee finance dataset proces"""
import os
import sys
import json
from utils import read_by_lines, write_by_lines, text_to_sents2, cal_md5
from utils import *
from constants import replaceSpace

enum_role = "环节"
max_trigger_len1 = 0
max_role_len1= 0
long_trigger_sentence = 0
long_role_sentence = 0


def data_process(path, model="trigger", is_predict=False):
    """data_process"""

    def label_data(data, start, l, _type):
        """label_data"""
        for i in range(start, start + l):
            suffix = "B-" if i == start else "I-"
            data[i] = "{}{}".format(suffix, _type)
        return data

    sentences = []
    output = ["text_a"] if is_predict else ["text_a\tlabel"]

    for line in read_by_lines(path):
        d_json = json.loads(line)
        _id = d_json["id"]
        text_a = [
            "，" if t == " " or t == "\n" or t == "\t" else t
            for t in list(d_json["text"].lower())
        ]
        if is_predict:
            sentences.append({"text": d_json["text"], "id": _id})
            output.append('\002'.join(text_a))
        else:
            if model == u"trigger":
                global max_trigger_len1,long_trigger_sentence
                labels = ["O"] * len(text_a)
                if len(d_json.get("event_list", [])) == 0:
                    continue
                if max_trigger_len1 < len(text_a):
                    max_trigger_len1 = len(text_a)
                for event in d_json.get("event_list", []):
                    event_type = event["event_type"]
                    start = event["trigger_start_index"]
                    trigger = event["trigger"]
                    labels = label_data(labels, start, len(trigger), event_type)
                output.append("{}\t{}".format('\002'.join(text_a), '\002'.join(
                    labels)))
            elif model == u"role":
                global  max_role_len1,long_role_sentence
                if max_role_len1<len(text_a):
                    max_role_len1 = len(text_a)

                for event in d_json.get("event_list", []):
                    labels = ["O"] * len(text_a)
                    for arg in event["arguments"]:
                        role_type = arg["role"]
                        if role_type == enum_role:
                            continue
                        if max_trigger_len1 < len(text_a):
                            max_trigger_len1 = len(text_a)
                        argument = arg["argument"]
                        start = arg["argument_start_index"]
                        labels = label_data(labels, start,
                                            len(argument), role_type)
                    output.append("{}\t{}".format('\002'.join(text_a),
                                                  '\002'.join(labels)))
    print(max_trigger_len1,max_role_len1)
    return output


def enum_data_process(path, is_predict=False):
    """enum_data_process"""
    output = ["text_a"] if is_predict else ["label\ttext_a"]
    for line in read_by_lines(path):
        d_json = json.loads(line)
        text = d_json["text"].lower().replace("\t", " ")
        text = [
            "，" if t == " " or t == "\n" or t == "\t" else t
            for t in list(d_json["text"].lower())
        ]
        if is_predict:
            output.append('\002'.join(text))
            continue
        if len(d_json.get("event_list", [])) == 0:
            continue
        label = None
        for event in d_json["event_list"]:
            if event["event_type"] != "公司上市":
                continue
            for argument in event["arguments"]:
                role_type = argument["role"]
                if role_type == enum_role:
                    label = argument["argument"]
        if label:
            output.append("{}\t{}".format(label, '\002'.join(text)))
    return output


def schema_process(path, model="trigger"):
    """schema_process"""

    def label_add(labels, _type):
        """label_add"""
        if "B-{}".format(_type) not in labels:
            labels.extend(["B-{}".format(_type), "I-{}".format(_type)])
        return labels

    labels = []
    for line in read_by_lines(path):
        d_json = json.loads(line.strip())
        if model == "trigger":
            labels = label_add(labels, d_json["event_type"])
        elif model == "role":
            for role in d_json["role_list"]:
                if role["role"] == enum_role:
                    continue
                labels = label_add(labels, role["role"])
        elif model == "enum":
            for role in d_json["role_list"]:
                if role["role"] == enum_role:
                    labels = role["enum_items"]

    labels.append("O")
    tags = []
    for index, label in enumerate(labels):
        tags.append("{}\t{}".format(index, label))
    if model == "enum":
        tags = tags[:-1]
    return tags


def marked_doc_2_sentence(doc):
    """marked_doc_2_sentence"""

    def argument_in_sent(sent, argument_list, trigger):
        """argument_in_sent"""
        trigger_start = sent.find(trigger)
        if trigger_start < 0:
            return trigger_start, [], None
        new_arguments, enum_argument = [], None
        for argument in argument_list:
            word = argument["argument"]
            word = word.replace(" ",replaceSpace)
            role_type = argument["role"]
            if role_type == enum_role:
                # special
                enum_argument = argument
                continue
            start = sent.find(word)
            if start < 0:
                continue
            new_arguments.append({
                "role": role_type,
                "argument": word,
                "argument_start_index": start
            })
        return trigger_start, new_arguments, enum_argument

    title = doc["title"]
    title = text_to_title2(title)
    if len(title)>0:
        if title[-1] not in ['。','？','！']:
            title.append('。')
    text = doc["text"]
    sents = text_to_sents2(text)
    exist_sents, sent_mapping_event, sents_order = set(), {}, []
    step = 3
    ## original split sentences methods
    # batch_sents = [sents[i:i + step] for i in range(0, len(sents), step)]
    ## edit split sentences methods, form[123,456,789,...] to [123,234,345,...]
    batch_sents = []
    if len(sents)<step:
        batch_sents = [sents[i:i + step] for i in range(0, len(sents), step)]
    else:
        for i in range(0,len(sents)-(step-1)):
            eachsent = sents[i:i+step]
            batch_sents.append(eachsent)

    if len(title)>0 and len(batch_sents)>0:
        if len(batch_sents[0][0])>4 and batch_sents[0][0][0:4] == '原标题：':
            begin_index = 1
        else:
            begin_index = 0
        for i in range(begin_index, len(batch_sents)):
            title=["".join(title)]
            batch_sents[i] = title + batch_sents[i]
            # batch_sents[i] = [" ".join(title + batch_sents[i])]

    # if len(title) > 0:
    #     batch_sents = [[title]] + batch_sents
    for batch in batch_sents:
        # b_sent = " ".join(batch).replace("\n", " ").replace(
        #     "\r\n", " ").replace("\r", " ").replace("\t", " ")
        b_sent = "".join(batch).replace("\n", replaceSpace).replace(
            "\r\n", replaceSpace).replace("\r", replaceSpace).replace("\t", replaceSpace)
        if b_sent in sent_mapping_event:
            continue
        sent_id = cal_md5(b_sent.encode("utf-8"))
        sent_mapping_event[b_sent] = {
            "id": doc["id"],
            "sent_id": sent_id,
            "text": b_sent,
            "length":len(b_sent)
        }
        sents_order.append(b_sent)

    for event in doc.get("event_list", []):
        for sent in sents_order:
            tri_start, argus, enum_arg = -1, [], None
            tri_start, argus, enum_arg = argument_in_sent(
                sent, event["arguments"], event["trigger"])
            if tri_start < 0:
                continue
            # if len(argus) > len(arguments):
            #     cur_sent, trigger_start, arguments = sent, tri_start, argus
            if tri_start >= 0 and len(argus) > 0:
                # add enum 2 event
                if enum_arg:
                    argus.append(enum_arg)
                if "event_list" not in sent_mapping_event[sent]:
                    sent_mapping_event[sent]["event_list"] = []
                new_event = {
                    "arguments": argus,
                    "event_type": event["event_type"],
                    "trigger": event["trigger"],
                    "trigger_start_index": tri_start
                }
                sent_mapping_event[sent]["event_list"].append(new_event)
    # ttttt = sent_mapping_event
    # dddd = sent_mapping_event.values()
    # print(sent_mapping_event)
    # print("-----")

    return sent_mapping_event.values()


def docs_data_process(path):
    """docs_data_process"""
    lines = read_by_lines(path)
    sentences = []
    for line in lines:
        d_json = json.loads(line)
        sentences.extend(marked_doc_2_sentence(d_json))
    sentences = [json.dumps(s, ensure_ascii=False) for s in sentences]
    return sentences


if __name__ == "__main__":
    # schema process
    print("\n=================DUEE FINANCE DATASET==============")
    conf_dir = "./conf/DuEE-Fin"
    schema_path = "{}/event_schema.json".format(conf_dir)
    tags_trigger_path = "{}/trigger_tag.dict".format(conf_dir)
    tags_role_path = "{}/role_tag.dict".format(conf_dir)
    tags_enum_path = "{}/enum_tag.dict".format(conf_dir)
    print("\n=================start schema process==============")
    print('input path {}'.format(schema_path))
    tags_trigger = schema_process(schema_path, "trigger")   #["id\tB-事件类型"]
    write_by_lines(tags_trigger_path, tags_trigger)
    print("save trigger tag {} at {}".format(
        len(tags_trigger), tags_trigger_path))
    tags_role = schema_process(schema_path, "role")   #["id\tB-role"]
    write_by_lines(tags_role_path, tags_role)
    print("save trigger tag {} at {}".format(len(tags_role), tags_role_path))
    tags_enum = schema_process(schema_path, "enum")  #[id\t筹备上市]空
    write_by_lines(tags_enum_path, tags_enum)
    print("save enum enum tag {} at {}".format(len(tags_enum), tags_enum_path))
    print("=================end schema process===============")

    # data process
    data_dir = "./data1/DuEE-Fin"
    sentence_dir = "{}/sentence".format(data_dir)
    trigger_save_dir = "{}/trigger".format(data_dir)
    role_save_dir = "{}/role".format(data_dir)
    enum_save_dir = "{}/enum".format(data_dir)
    print("\n=================start data process==============")
    print("\n********** start document process **********")
    if not os.path.exists(sentence_dir):
        os.makedirs(sentence_dir)
    train_sent = docs_data_process("{}/train.json".format(data_dir))
    write_by_lines("{}/train.json".format(sentence_dir), train_sent)
    dev_sent = docs_data_process("{}/dev.json".format(data_dir))
    write_by_lines("{}/dev.json".format(sentence_dir), dev_sent)
    test_sent = docs_data_process("{}/test1.json".format(data_dir))
    write_by_lines("{}/test1.json".format(sentence_dir), test_sent)
    print("train {} dev {} test1 {}".format(
        len(train_sent), len(dev_sent), len(test_sent)))
    print("********** end document process **********")

    print("\n********** start sentence process **********")
    print("\n----trigger------for dir {} to {}".format(sentence_dir,
                                                       trigger_save_dir))
    if not os.path.exists(trigger_save_dir):
        os.makedirs(trigger_save_dir)
    train_tri = data_process("{}/train.json".format(sentence_dir), "trigger")
    write_by_lines("{}/train.tsv".format(trigger_save_dir), train_tri)
    dev_tri = data_process("{}/dev.json".format(sentence_dir), "trigger")
    write_by_lines("{}/dev.tsv".format(trigger_save_dir), dev_tri)
    test_tri = data_process("{}/test1.json".format(sentence_dir), "trigger")
    write_by_lines("{}/test1.tsv".format(trigger_save_dir), test_tri)
    print("train {} dev {} test1 {}".format(
        len(train_tri), len(dev_tri), len(test_tri)))

    print("\n----role------for dir {} to {}".format(sentence_dir,
                                                    role_save_dir))
    if not os.path.exists(role_save_dir):
        os.makedirs(role_save_dir)
    train_role = data_process("{}/train.json".format(sentence_dir), "role")
    write_by_lines("{}/train.tsv".format(role_save_dir), train_role)
    dev_role = data_process("{}/dev.json".format(sentence_dir), "role")
    write_by_lines("{}/dev.tsv".format(role_save_dir), dev_role)
    test_role = data_process("{}/test1.json".format(sentence_dir), "role")
    write_by_lines("{}/test1.tsv".format(role_save_dir), test_role)
    print("train {} dev {} test1 {}".format(
        len(train_role), len(dev_role), len(test_role)))

    print("\n----enum------for dir {} to {}".format(sentence_dir,
                                                    enum_save_dir))
    if not os.path.exists(enum_save_dir):
        os.makedirs(enum_save_dir)
    trian_enum = enum_data_process("{}/train.json".format(sentence_dir))
    write_by_lines("{}/train.tsv".format(enum_save_dir), trian_enum)
    dev_enum = enum_data_process("{}/dev.json".format(sentence_dir))
    write_by_lines("{}/dev.tsv".format(enum_save_dir), dev_enum)
    test_enum = enum_data_process("{}/test1.json".format(sentence_dir))
    write_by_lines("{}/test1.tsv".format(enum_save_dir), test_enum)
    print("train {} dev {} test1 {}".format(
        len(trian_enum), len(dev_enum), len(test_enum)))
    print("********** end sentence process **********")
    print("=================end data process==============")
