import json
from constants import replaceSpace
path = '/home/xuwd/projects/DuEE-pytorch/data/DuEE-Fin/sentence/test1.json'

list = []
with open(path,'r',encoding='utf-8') as f:
    for line in f:
        line = json.loads(line)
        id = line.get("id")
        if id not in list:
            list.append(id)
print(len(list))




# path = '/home/xuwd/projects/DuEE-pytorch/data/DuEE-Fin/train.json'
# writepath = '/home/xuwd/projects/DuEE-pytorch/data1/DuEE-Fin/save_liu/dev.json'
#
# import torch
#
# x=torch.arange(24).view(4,3,2)
# x=x.float()
# print(x)
# b = x.mean(dim=1)
# print(b)
# print(b.size())


import re

# def find_all_index(word, sent):
#     wordlength = len(word) #aa
#     sentlength = len(sent) #aabbccaa
#     ret = []
#     for i in range(0,sentlength-wordlength+1):
#         if sent[i:i+wordlength]==word:
#             ret.append(i)
#     return ret


# somestr = 'sldjkhfkdhfslddfhdhf'
# sub = 'sld'
# all_index = find_all_index(sub,somestr) # 得到所有下标[6, 18, 27, 34, 39]
# print(all_index)

# num1 = 0
# with open(path,'r',encoding='utf-8') as f:
#     with open(writepath,'w',encoding='utf-8') as wr:
#         for line in f:
#             dict = {}
#             d_json = json.loads(line)
#             # for key, value in d_json.items():
#             #     print(key)
#             # print(d_json)
#             id = d_json['id']
#             sent_id = d_json['sent_id']
#             text = d_json['text']
#             length = d_json['length']
#             if length>512:
#                 # print(text,length)
#                 num1+=1
#             event_list = d_json.get('event_list',[])
#             if event_list==[]:
#                 continue
#             else:
#                 dict['id'] = id
#                 dict['sent_id'] = sent_id
#                 dict['text'] = text
#                 dict['event_list'] = event_list
#                 dict['length'] = length
#                 ddata = json.dumps(dict, ensure_ascii=False)
#                 wr.write(ddata)
#                 wr.write('\n')
#             # break
# print(num1)
"""
# path = '/home/xuwd/projects/DuEE-pytorch/data/DuEE-Fin/sentence/train.json'
num=0
dict = {1:0,2:0,3:0,4:0,5:0}
with open(path,'r',encoding='utf-8') as f:
    for line in f:
        d_json = json.loads(line)
        texts = d_json['text']
        # print(d_json)
        # print(texts)
        # print("----")
        id = d_json['id']
        # print(id)
        # for text in texts:
        #     if len(text)>0:
        #         c = text[-1]
        #         if c not in dict.keys():
        #             dict[c] = 1
        #         else:
        #             dict[c] += 1
        for event in d_json.get('event_list',[]):
            # print(event)
            # print("****")
            argu =  event['arguments']
            if event==[]:
                continue
            for  each in argu:
                role = each['role']
                argum = each['argument']
                xxx=replaceSpace
                space  = argum.find(xxx)
                spacenum = 0
                if space>0:
                    num+=1
                    # print(id,texts,argum)
                # if space>0:
                #     num += 1
                #     for i in range(0,len(argum)-1):
                #         if i==replaceSpace:
                #             spacenum+=1
                #     if spacenum>0:
                #         if spacenum==2:
                #             print(id,texts,argum)
                #         if spacenum not in dict.keys():
                #             dict[spacenum] = 1
                #         else:
                #             dict[spacenum] += 1


b = sorted(dict.items(), key=lambda item:item[1])
print(b)

print(num)
# #         break
# # print(dict)
# b = sorted(dict.items(), key=lambda item:item[1])
# print(b)
"""
