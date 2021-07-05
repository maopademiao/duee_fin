import json
import argparse

parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--oripath", type=str, default='/home/xuwd/projects/DuEE-pytorch/submit/new1/duee-fin55.61.json', help="original result path")
parser.add_argument("--rightpath", type=str, default='/home/xuwd/projects/DuEE-pytorch/submit/new1/duee-fin-55.61v2.json', help="writepath")
parser.add_argument("--fullidpath", type=str, default='/home/xuwd/projects/DuEE-pytorch/submit/new1/duee-fin-spanner-56.02.json', help="writepath")



args = parser.parse_args()

fullidpath = args.fullidpath
oripath = args.oripath
rightpath = args.rightpath

list = []
with open(fullidpath,'r',encoding='utf-8') as f1:
    for lines in f1:
        line = json.loads(lines)
        id = line.get('id')
        list.append(id)
print(len(list))

datadict = {}
with open(oripath,'r',encoding='utf-8') as f3:
    for lines in f3:
        line = json.loads(lines)
        id = line.get('id')
        datadict[id] = line.get('event_list')

print(len(datadict))
# print(datadict)

wf = open(rightpath,'w',encoding='utf-8')

for i in list:
    if datadict.get(i,0)==0:
        print(i)
        continue
    else:
        dict = {}
        dict['id'] = i
        dict['event_list'] = datadict[i]
        wf.write(json.dumps(dict, ensure_ascii=False) + '\n')

wf.close()

