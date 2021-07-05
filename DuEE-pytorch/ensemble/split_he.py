import json
import argparse

parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--originalpath", type=str, default='/home/xuwd/projects/DuEE-pytorch/submit/new2/duee-fin.json', help="original result path")
parser.add_argument("--splitpath", type=str, default='/home/xuwd/projects/DuEE-pytorch/submit/new2/duee-fin-split-he.json', help="writepath")


args = parser.parse_args()

splitpath = args.splitpath
originalpath = args.originalpath

not_split_helist = ['股份有限公司','物资储备局','城市建设部','新疆维吾尔自治区','丰投资','老恒','科技','证券',
                    '国','中华人民共','集团','60%','2020年9月2日','邦生物科技股份有限公司','信永中','新希望六','的监事会主席肖大波',
                    '雄研院业务合作支撑-售前支撑服务采购项目','深圳市住房','6%','包头天','成都仁','100%股权及相应权利',
                    '贝朗医','新疆生产建设兵团住房','电子科技集团股份有限公司','江苏润','英大泰','广东省江门市新会区主城区环卫一体化',
                    '架空绝缘导线','试点建设-设计开发”项目','所有流通A类股','河源市','北京正','上海养','上海谊','功能材料业务',
                    '上海锦','欧洲经济','国家粮食','浙江省住房','州债券','亿嘉','国家安全教研室主任','成股份有限公司','2020年-2021年中移集成',
                    '薪酬委员会委员','新建富馀煤气','隆基泰','磐茂投资','平科技股份有限公司','旅游部党组书记','苏州协','招标采购管理服务网',
                    '广东江门新会区主城区环卫一体化','8月4','盛杰食品有限公司','丰投资股份有限公司','辉光电股份(11',
                    '盈同创投资顾问有限公司','城乡建设部','记黄埔将持有合并公司约15.7%的股份','俊与中车长客公司','成股份有限公司',
                    '基础设施部长','7月25日','烟台泰','众创业投资合伙企业（有限合伙）','科达精密清洗设备股份有限公司','规划局',
                    '城乡建设局','573号改扩建项目','工业','信息化委员会办公室','上海市住房','首席战略','改革委员会','深圳瑞','亿嘉']

singlecharlist = ['谐','邦','丰','氏','悦','瑞','康']
beforechar = ['呼','协','仁','共','同','利']

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
                elif '和' in argument:
                    argument1 = argument.split('和')
                    flag = 0
                    if len(argument1)==2:
                        argum1 = argument1[0]
                        argum2 = argument1[1]
                        if len(argum1)<=1 or len(argum2)<=1:
                            flag = 1
                        elif len(argum2)>1 and argum2[0] in singlecharlist:
                            # print(argum1, "|", argum2)
                            flag = 1
                        elif len(argum1)>1 and argum1[-1] in beforechar:
                            flag = 1
                        if flag == 0:
                            # print(argum1,"|",argum2)
                            if argum1 in not_split_helist or argum2 in not_split_helist:
                                flag = 1
                            else:
                                # print(argum1, "|||", argum2)
                                for key in argument1:
                                    dict1 = {}
                                    dict1['role'] = role
                                    dict1['argument'] = key
                                    neweacharguments.append(dict1)
                        # elif flag == 0 and abs(len(argum1)-len(argum2))>=1:
                        #     print(argum1, "|", argum2)
                        #     pass
                    else:
                        dict1 = {}
                        dict1['role'] = role
                        dict1['argument'] = argument
                        neweacharguments.append(dict1)
                    if flag == 1:
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