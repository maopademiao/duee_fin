import json
import argparse

parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--fullidpath", type=str, default='/home/xuwd/projects/DuEE-pytorch/submit/new1/duee-fin-spanner-56.02.json', help="original result path")
parser.add_argument("--lessidpath", type=str, default='/home/xuwd/projects/DuEE-pytorch/submit/new2/zy.json', help="writepath")

parser.add_argument("--addidpath", type=str, default='/home/xuwd/projects/DuEE-pytorch/submit/new2/zy3w.json', help="writepath")


args = parser.parse_args()

fullidpath = args.fullidpath
lessidpath = args.lessidpath
addidpath = args.addidpath

list = []
with open(fullidpath,'r',encoding='utf-8') as f1:
    for lines in f1:
        line = json.loads(lines)
        id = line.get('id')
        list.append(id)


datadict = {}
with open(lessidpath,'r',encoding='utf-8') as f3:
    for lines in f3:
        line = json.loads(lines)
        id = line.get('id')
        datadict[id] = line.get('event_list')



wf = open(addidpath,'w',encoding='utf-8')

for i in list:
    if datadict.get(i,[])==[]:
        dict = {}
        dict['id'] = i
        dict['event_list'] = []
        wf.write(json.dumps(dict, ensure_ascii=False) + '\n')
    else:
        dict = {}
        dict['id'] = i
        dict['event_list'] = datadict[i]
        wf.write(json.dumps(dict, ensure_ascii=False) + '\n')

wf.close()

