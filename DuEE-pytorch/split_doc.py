import json
from utils import text_to_sents
import xlwt




def split_doc(read_path, write_path):

    f = open(write_path, 'w', encoding='UTF-8')
    with open(read_path) as f1:

        for line in f1:

            d_json = json.loads(line.strip())
            event_list = d_json.get('event_list', [])
            if event_list==[]:continue
            sents = text_to_sents(d_json['text'])
            sents = [d_json['title']] + sents
            sents = ["".join(sent.split(" ")) for sent in sents]

            for event in event_list:

                exist_sents=[]
                for value in event['arguments']:
                    if value['role'] =="环节":
                        continue
                    s=[]
                    for i in range(len(sents)):
                        if "".join(value['argument'].split(" ")) in sents[i]:
                            s.append(i)
                    exist_sents.append(s)
                for i in range(len(sents)):
                    if "".join(event['trigger'].split(" ")) in sents[i]:
                        s.append(i)
                exist_sents.append(s)
                print(exist_sents)
                res = split_event(exist_sents)
                min_res = min(res)
                max_res = max(res)
                new_doc = sents[min_res : max_res+1]


                new_documents = {'id':d_json['id'], 'text': new_doc, 'event': event, 'title': d_json['title']}
                f.write(json.dumps(new_documents, ensure_ascii=False) +'\n')
            #print(new_documents)


def split_event(exist_sents):
    res = []
    while exist_sents !=[]:
        statistic = {}
        for x in exist_sents:
            for i in range(len(x)):
                if x[i] not in statistic:
                    statistic[x[i]] = 1

                else:
                    statistic[x[i]] += 1
        #print(statistic,exist_sents)
        m = max(statistic.values())

        for key, value in statistic.items():
            if value == m:
                res.append(key)
        c = []
        for s in exist_sents:
            if not any(i in s for i in res):
                c.append(s)
        exist_sents = c
    return res

def length_distribution(path):
    length = {}
    f1 = open('./processed/大于30的句子.json', 'w', encoding='UTF-8')
    with open(path) as f:
        for l in f:
            text = json.loads(l)['text']
            if len(text)>=30:
                f1.write(json.dumps(json.loads(l), ensure_ascii=False) + '\n')
            if len(text) not in length:
                length[len(text)] = 1
            else:
                length[len(text)] +=1
    length = list(length.items())
    length = sorted(length, key = lambda x:x[0])
    writeExcel(length)
    print(length)

def writeExcel(length):
    writebook = xlwt.Workbook()
    test = writebook.add_sheet('distribution')
    test.write(0, 0,"事件跨句长度")
    test.write(0, 1, "数量")
    test.write(0, 2, "所占比例")
    all_num = sum([v[1] for v in length])

      # Change model to 'eval' mode.
    num = 1

    for a,b in length:
        test.write(num,0,a)
        test.write(num,1,b)
        test.write(num,2,float(b/all_num))
        num+=1
    writebook.save('./processed/span_statistic.xls')



if __name__ == '__main__':
    split_doc('./data/DuEE-Fin/train.json', './processed/train.json')
    length_distribution('./processed/train.json')