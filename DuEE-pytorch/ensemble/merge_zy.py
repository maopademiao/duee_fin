import json
import argparse
# 将两个结果简单合并

parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--xupath", type=str, default='/home/xuwd/projects/DuEE-pytorch/submit/new2/last-ner.json', help="original result path")
parser.add_argument("--writepath", type=str, default='/home/xuwd/projects/DuEE-pytorch/submit/new2/duee-fin-tmp.json', help="writepath")
parser.add_argument("--zypath", type=str, default='/home/xuwd/projects/DuEE-pytorch/submit/new2/test2.all_cpt_all_ids-split.json', help="writepath")

# parser.add_argument("--zypath", type=str, default='/home/xuwd/projects/DuEE-pytorch/submit/new1/duee-fin-39zy56.53-addid.json', help="writepath")


args = parser.parse_args()

xupath=args.xupath
zypath=args.zypath
writepath= args.writepath

wf = open(writepath,'w',encoding='utf-8')

with open(xupath,'r',encoding='utf-8') as f1, open(zypath,'r',encoding='utf-8') as f2:
    for line1,line2 in zip(f1,f2):
        line1 = json.loads(line1)
        line2 = json.loads(line2)
        id1 = line1.get('id')
        id2 = line2.get('id')
        assert id1==id2
        savedict = {}
        savedict['id'] = id1
        eventlist1 = line1.get('event_list')
        eventlist2 = line2.get('event_list')
        eventlist = []
        for key in eventlist1:
            eventlist.append(key)
        for key in eventlist2:
            eventlist.append(key)
        savedict['event_list'] = eventlist
        wf.write(json.dumps(savedict, ensure_ascii=False) + '\n')
        # print(line2)
        # print(savedict)
        # break
wf.close()




