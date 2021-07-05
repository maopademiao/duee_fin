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

import hashlib
from constants import replaceSpace, max_sent_length


def cal_md5(str):
    """calculate string md5"""
    str = str.decode("utf-8", "ignore").encode("utf-8", "ignore")
    return hashlib.md5(str).hexdigest()


def read_by_lines(path):
    """read the data by line"""
    result = list()
    with open(path, "r") as infile:
        for line in infile:
            result.append(line.strip())
    return result


def write_by_lines(path, data):
    """write the data"""
    with open(path, "w") as outfile:
        newdata = []
        # for d in data:
        #     newdata.append(d)
        # outfile.write()
        [outfile.write(d + "\n") for d in data]


def text_to_sents(text):
    """text_to_sents"""
    deliniter_symbols = [u"。", u"？", u"！"]
    paragraphs = text.split("\n")
    ret = []
    for para in paragraphs:
        if para == u"":
            continue
        sents = [u""]
        for s in para:
            sents[-1] += s
            if s in deliniter_symbols:
                sents.append(u"")
        if sents[-1] == u"":
            sents = sents[:-1]
        ret.extend(sents)
    print("ret",ret)
    print("\n")
    return ret

def text_to_title2(text):
    """text_to_sents"""
    # deliniter_symbols = [u"。", u"？", u"！"]
    # print(text)
    text = text.replace('\n',' ')
    endl_symbols = ['。','：','】','.','■','；','？','！','）',')','”','%',]
    newtext = ""
    for i, chara in enumerate(text):
        if chara=='\n':
            if i==0:
                continue
            if text[i-1] in endl_symbols:
                continue
            else:
                newtext+=','
        elif chara==' ':
            # newtext+=replaceSpace
            if i==0:
                newtext += replaceSpace
            elif text[i-1]==' ':
                continue
            else:
                newtext += replaceSpace
        else:
            newtext+=chara
    paragraphs = newtext
    # print("----")
    # print(paragraphs)
    ret = []
    for para in paragraphs:
        ret.append(para)
    return ret

def text_to_sents2(text):
    """text_to_sents"""
    deliniter_symbols = [u"。", u"？", u"！"]
    endl_symbols = ['。', '：', '】', '.', '■', '；', '？', '！', '）', ')', '”', '%', ]
    # print(text)
    newtext = ""
    for i, chara in enumerate(text):
        if chara=='\n':
            if i==0:
                continue
            if text[i-1] in endl_symbols:
                continue
            else:
                newtext+=','
        elif chara==' ':
            # newtext+=replaceSpace
            if i==0:
                newtext += replaceSpace
            if text[i-1]==' ':
                continue
            else:
                newtext += replaceSpace

        else:
            newtext+=chara
    ret = []
    sents = ""
    for chara in newtext:
        sents += chara
        if chara in deliniter_symbols:
            if len(sents)>20:
                ret.append(sents)
                sents=""
    if len(sents)!=0:
        ret.append(sents)
    # print(ret)
    # print("---")
    return ret


def text_to_sents4(text):
    """text_to_sents"""
    deliniter_symbols = [u"。", u"？", u"！"]
    endl_symbols = ['。', '：', '】', '.', '■', '；', '？', '！', '）', ')', '”', '%', ]
    # print(text)
    newtext = ""
    for i, chara in enumerate(text):
        if chara=='\n':
            if i==0:
                continue
            if text[i-1] in endl_symbols:
                continue
            else:
                newtext+=','
        elif chara==' ':
            # newtext+=replaceSpace
            if i==0:
                newtext += replaceSpace
            if text[i-1]==' ':
                continue
            else:
                newtext += replaceSpace

        else:
            newtext+=chara
    ret = []
    sents = ""
    for i,chara in enumerate(newtext):
        sents += chara
        if chara in deliniter_symbols:
            if len(sents)>20:
                ret.append(sents)
                # print(len(sents),sents)
                sents=""
    if len(sents)!=0:
        ret.append(sents)
    # print(ret)
    newret = []
    for eachsent in ret:
        if len(eachsent)>max_sent_length:
            aa = get_middle_douhao_position(eachsent)
            for i in range(0,len(aa)):
                if len(aa[i])>max_sent_length:
                    ab = get_middle_douhao_position(aa[i])
                    newret+=ab
                else:
                    newret.append(aa[i])
        else:
            newret.append(eachsent)
    # for i in newret:
    #     print(len(i),i)


    # print("---")
    return newret


def get_middle_douhao_position(sent):
    split_sent_num = len(sent)//max_sent_length+1
    ret = []
    # if split_sent_num==2:
    middle_position = len(sent)//2
    sss = sent[:middle_position]
    sent1 = sss[::-1]
    sent2 = sent[middle_position:len(sent)]
    start1 = sent1.find('，')
    if start1<0:
        start1 = sent1.find(',')
    # print(sent1)
    # print(type(sent1))
    start2 = sent2.find('，')
    if start2<0:
        start2 = sent2.find(',')
    if start1<0 and start2<0:
        ret.append(sent)
        # print("无逗",len(sent), sent)
    elif start1<0 and start2>=0:
        # print("处理1",len(sent))
        newsent1 = sent1[::-1]+sent2[:start2+1]
        newsent2 = sent2[start2+1:]
        ret.append(newsent1)
        ret.append(newsent2)
    elif start1>=0 and start2<0:
        # print("处理2",len(sent))
        ddd1 = sent1[start1:len(sent1)]
        newsent1 = ddd1[::-1]
        ddd2 = sent1[0:start1]
        newsent2 = ddd2[::-1] + sent2
        ret.append(newsent1)
        ret.append(newsent2)
    else:
        # print("处理3",len(sent))
        if start1<start2:
            ddd1 = sent1[start1:len(sent1)]
            newsent1 = ddd1[::-1]
            ddd2 = sent1[0:start1]
            newsent2 = ddd2[::-1] + sent2
            ret.append(newsent1)
            ret.append(newsent2)
        else:
            newsent1 = sent1[::-1] + sent2[:start2 + 1]
            newsent2 = sent2[start2 + 1:]
            ret.append(newsent1)
            ret.append(newsent2)
    return ret

def load_dict(dict_path):
    """load_dict"""
    vocab = {}
    for line in open(dict_path, 'r', encoding='utf-8'):
        value, key = line.strip('\n').split('\t')
        vocab[key] = int(value)
    return vocab


def extract_result(text, labels):
    """extract_result"""
    ret, is_start, cur_type = [], False, None
    aa,ab = len(text),len(labels)
    if len(text) != len(labels):
        # 韩文回导致label 比 text要长
        labels = labels[:len(text)]
    for i, label in enumerate(labels):
        if label != u"O":
            _type = label[2:]
            if label.startswith(u"B-"):
                is_start = True
                cur_type = _type
                ret.append({"start": i, "text": [text[i]], "type": _type})
            elif _type != cur_type:
                """
                # 如果是没有B-开头的，则不要这部分数据
                cur_type = None
                is_start = False
                """
                cur_type = _type
                is_start = True
                ret.append({"start": i, "text": [text[i]], "type": _type})
            elif is_start:
                ret[-1]["text"].append(text[i])
            else:
                cur_type = None
                is_start = False
        else:
            cur_type = None
            is_start = False
    return ret


if __name__ == "__main__":
    s = "xxdedewd"
    print(cal_md5(s.encode("utf-8")))
