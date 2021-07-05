import json
import argparse
# 分割中文逗号，前一个字符是数字不动，不是数字分开

parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--originalpath", type=str, default='/home/xuwd/projects/DuEE-pytorch/submit/new2/last-duee-fin-splitdou.json', help="original result path")
parser.add_argument("--splitpath", type=str, default='/home/xuwd/projects/DuEE-pytorch/submit/new2/last-duee-fin-splitdou-v2.json', help="writepath")

args = parser.parse_args()

splitpath = args.splitpath
originalpath = args.originalpath

digit = ['0','1','2','3','4','5','6','7','8','9']
f = open(splitpath,'w',encoding='utf-8')
with open(originalpath,'r',encoding='utf-8') as rf:
    for lines in rf:
        line = json.loads(lines)
        id = line.get('id')
        event_list = line.get('event_list')
        savedict = {}
        savedict['id'] = id
        # print(id)
        neweventlist = []
        for event in event_list:
            event_type = event['event_type']
            arguments = event['arguments']
            if len(arguments)==0:
                continue
            neweachevent = {}
            neweachevent['event_type'] = event_type
            neweacharguments = []
            for eacharg in arguments:
                role = eacharg['role']
                argument = eacharg['argument']
                flag = 0
                if argument==None:
                    continue

                elif ',' in argument:
                    argument1 = argument.split(',')

                    for i in range(0,len(argument1)-1):
                        key = argument1[i]
                        if len(key)>0:
                            if 'a'<= key[-1]<='z' or 'A'<=key[-1]<='Z':
                                flag = 2
                                break
                            if key[-1] in digit:
                                flag = 1
                                break
                    if flag == 0:
                        for i in range(0, len(argument1) - 1):
                            key = argument1[i]
                            if len(key)==0:
                                continue
                            else:
                                dict1 = {}
                                dict1['role'] = role
                                dict1['argument'] = key
                                neweacharguments.append(dict1)
                    if flag==1 or flag==2:
                        dict1 = {}
                        dict1['role'] = role
                        dict1['argument'] = argument
                        neweacharguments.append(dict1)
                else:
                    dict1 = {}
                    dict1['role'] = role
                    dict1['argument'] = argument
                    neweacharguments.append(dict1)
            neweachevent['arguments'] = neweacharguments
            neweventlist.append(neweachevent)
        savedict['event_list'] = neweventlist
        f.write(json.dumps(savedict, ensure_ascii=False) + '\n')
        # break

f.close()