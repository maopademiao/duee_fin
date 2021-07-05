import json
import argparse

parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--originalpath", type=str, default='/home/xuwd/projects/DuEE-pytorch/submit/new2/duee-fin-split-he.json', help="original result path")
parser.add_argument("--splitpath", type=str, default='/home/xuwd/projects/DuEE-pytorch/submit/new2/duee-fin-split-he-dele..json', help="writepath")

args = parser.parse_args()

splitpath = args.splitpath
originalpath = args.originalpath


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
                if argument==None:
                    continue

                elif '。' in argument:
                    argument1 = argument.split('。')
                    for key in argument1:
                        if len(key)==0:
                            continue
                        dict1 = {}
                        dict1['role'] = role
                        dict1['argument'] = key
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