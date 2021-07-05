import json


max_len = 0
with open('./Data/meta-data/last-duee-fin-splitdou-delecopy.json', 'r') as f:

    for line in f:
        line = json.loads(line)
        id = line.get("id")
        events = line.get('event_list',[])
        if events ==[]:
            continue
        for event in line['event_list']:
            for argu in event['arguments']:
                #if 30<=len(argu['argument'])<=40:
                if "," in argu['argument']:
                    print( id, argu['argument'])
                max_len = max(max_len, len(argu['argument']))
print(max_len)