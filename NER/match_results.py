import json


import hashlib



def cal_md5(str):
    """calculate string md5"""
    str = str.decode("utf-8", "ignore").encode("utf-8", "ignore")
    return hashlib.md5(str).hexdigest()

path = '/home/xuwd/projects/multi-domain_NER-3.0/output/trigger/test2.txt'
write_path = '/home/xuwd/projects/multi-domain_NER-3.0/output/trigger/test_pred2.json'
tmp_path = '/home/xuwd/projects/multi-domain_NER-3.0/output/trigger/tmp2.json'

copy_path = '/home/xuwd/projects/DuEE-pytorch/data/DuEE-Fin/sentence/test2.json'

def write_tmp():
    tmpf = open(tmp_path,'w',encoding='utf-8')

    with open(path,'r',encoding='utf-8') as nerf:
        texts = []
        labels = []
        for eachner in nerf:
            if eachner=='\n':
                b_sent = "".join(texts)
                dict['sent_id']=cal_md5(b_sent.encode("utf-8"))
                dict['text'] = b_sent
                dict['labels'] = labels
                dict['length'] = len(texts)
                tmpf.write(json.dumps(dict, ensure_ascii=False) + '\n')
                texts = []
                labels =[]
                continue
            dict = {}
            eachner = eachner.strip('\n').split('\t')
            texts.append(eachner[0])
            labels.append(eachner[2])
            # print(eachner)
            # tmpf.write(json.dumps(dict, ensure_ascii=False) + '\n')
            # break
    tmpf.close()

write_tmp()

wf = open(write_path,'w',encoding='utf-8')

with open(tmp_path,'r',encoding='utf-8') as tmpf1, open(copy_path,'r',encoding='utf-8') as senf2:
    for x, y in zip(tmpf1, senf2):
        x = json.loads(x)
        y = json.loads(y)
        tmpsenid = x.get('sent_id')
        tmplength = x.get('length')

        orisentid = y.get('sent_id')
        orilength = y.get('length')
        assert tmpsenid==orisentid
        assert tmplength==orilength
        dict = {}
        dict['id'] = y.get('id')
        dict['sent_id'] = orisentid
        dict['text'] = y.get('text')
        dict['length'] = orilength
        dict['pred'] = {}
        dict['pred']['probs'] = [0]*orilength
        dict['pred']['labels'] = x.get('labels')
        wf.write(json.dumps(dict, ensure_ascii=False) + '\n')

        # print(x)
        # print(y)
        # break