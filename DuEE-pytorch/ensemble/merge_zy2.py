import json
import argparse
# 对结果去重

parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--path2", type=str, default='/home/xuwd/projects/DuEE-pytorch/submit/new2/last-duee-fin-splitdou-v2.json', help="original result path")
parser.add_argument("--writepath", type=str, default='/home/xuwd/projects/DuEE-pytorch/submit/new2/last-duee-fin-splitdou-delecopy.json', help="writepath")

args = parser.parse_args()

data = []
# path1 = '/home/xuwd/projects/DuEE-pytorch/submit/dee_eval.test.pred_span.Doc2EDAG.39.json'
path2 = args.path2

writepath = args.writepath



f = open(writepath,'w',encoding='utf-8')
with open(path2,'r',encoding='utf-8') as ftest:
    for line in ftest:
        line = json.loads(line)
        id = line.get('id')
        eventlist = line.get("event_list")
        savedict = {}
        savedict['id'] = id
        neweventlist = []
        existeventtype = []
        dict1 = {}
        saveargumentsdict = {}
        saveevent = {}
        for event in eventlist:
            event_type = event['event_type']
            arguments = event['arguments']
            if len(arguments)==0:
                continue
            if event_type not in existeventtype:
                # saveevent['event_type'] = event_type
                saveargumentsdict[event_type] = []
                existeventtype.append(event_type)
                dict1[event_type] = {}
            # dict = dict1[event_type]
            for eacharg in arguments:
                role = eacharg['role']
                argument = eacharg['argument']
                saveminidict = {}
                saveminidict['role'] = role
                saveminidict['argument'] = argument
                if argument in dict1[event_type].get(role,[]):
                    continue
                else:
                    saveargumentsdict[event_type].append(saveminidict)
                if dict1[event_type].get(role,[]) == []:
                    dict1[event_type][role] = []
                    dict1[event_type][role].append(argument)
                else:
                    dict1[event_type][role].append(argument)
        for each in existeventtype:
            saveevent = {}
            saveevent['event_type'] = each
            saveevent['arguments'] = saveargumentsdict[each]
            neweventlist.append(saveevent)
        savedict['event_list'] = neweventlist
        f.write(json.dumps(savedict, ensure_ascii=False) + '\n')
f.close()


# f = open(writepath,'w',encoding='utf-8')
# datadictzy = {}
# with open(path1,'r',encoding='utf-8') as rf:
#     for line in rf:
#         line = json.loads(line)
#         # print(line)
#         id = line.get("id")
#         eventlist = line.get("event_list")
#         datadictzy[id] = eventlist
#
# with open(path2,'r',encoding='utf-8') as rf:
#     for line in rf:
#         line = json.loads(line)
#         # print(line)
#         id = line.get('id')
#         savedict = {}
#         savedict['id'] = id
#         eventlist = line.get('event_list')
#         neweventlist = []
#         zyeventlist = datadictzy[id]
#         # if len(zyeventlist)>0 and len(eventlist)==0:
#         #     for k in zyeventlist:
#         #         neweventlist.append(k)
#         for event in eventlist:
#             dictevent = {}
#             event_type = event['event_type']
#             arguments = event['arguments']
#             if arguments==[]:
#                 continue
#             dict = {}
#             list = []
#             for zyevent in zyeventlist:
#                 if zyevent['event_type'] == event_type:
#                     for eachaaa in zyevent['arguments']:
#                         role = eachaaa['role']
#                         value = eachaaa['argument']
#                         dict[role] = value
#                         list.append(role)
#             for i, eacharg in enumerate(arguments):
#                 rolea = eacharg['role']
#                 if rolea in list:
#                     arguments[i]['argument'] = dict[rolea]
#             dictevent['event_type'] = event_type
#             dictevent['arguments'] = arguments
#             neweventlist.append(dictevent)
#         savedict['event_list'] = neweventlist
#         # print("---")
#         # print(savedict)
#         #
#         # break
#         f.write(json.dumps(savedict, ensure_ascii=False) + '\n')