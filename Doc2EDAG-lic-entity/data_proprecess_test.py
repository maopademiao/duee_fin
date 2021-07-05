import json
from dee.event_type import event_type_fields_list

def proprecess(path, sava_path):
    # with open('./Data/meta-data/event_schema.json', 'r', encoding='utf-8') as fc:
    #     for line in fc:
    #         line = json.loads(line)
    #         event2fields[line['event_type']] = []
    event2fields = dict(event_type_fields_list )
    all_examples = []
    with open(path,'r', encoding='utf-8') as f:

        for line in f:
            line = json.loads(line)
            #if 'event_list' not in line.keys():continue
            ann_valid_mspans = []
            ann_valid_dranges = []
            ann_mspan2dranges = {}
            ann_mspan2guess_field = {}
            recguid_eventname_eventdict_list = []
            #print(line)
            guid = line['id']
            sentences = line['text']
            for idx, event in enumerate(line.get('event_list',[])):

                event_type = event['event_type']
                fields = event2fields[event_type]
                cur_fields_span = {}
                all_fields_span = {}
                for argument in event['arguments']:
                    if argument['argument'] not in ann_valid_mspans:
                        ann_valid_mspans.append(argument['argument'])
                        ann_valid_dranges.extend(argument['index'])
                        ann_mspan2dranges[argument['argument']] = argument['index']
                        ann_mspan2guess_field[argument['argument']] = argument['role']
                    cur_fields_span[argument['role']] = argument['argument']
                for field in fields:
                    if field in cur_fields_span.keys():
                        all_fields_span[field] = cur_fields_span[field]
                    else:
                        all_fields_span[field] = None
                recguid_eventname_eventdict_list.append([idx, event_type ,all_fields_span])

            all_examples.append([guid, {'sentences': sentences, 'ann_valid_mspans': ann_valid_mspans, 'ann_valid_dranges': ann_valid_dranges, 'ann_mspan2dranges': ann_mspan2dranges,\
                                        'ann_mspan2guess_field': ann_mspan2guess_field, 'recguid_eventname_eventdict_list':recguid_eventname_eventdict_list}])
    fw = open(sava_path, 'w', encoding='utf-8')
    fw.write(json.dumps(all_examples ,indent=2, ensure_ascii=False))
    fw.close()
    print(len(all_examples))
















if __name__ == "__main__":
    #proprecess('./Data/meta-data/train.json', './Data/train.json')
    #proprecess('./Data/meta-data/dev.json', './Data/dev.json')
    proprecess('./Data/meta-data/test2.json', './Data/test2.json')