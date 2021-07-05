import pickle
import json
import numpy
from dee.event_type import event_type_fields_list



examples = {}
with open('./Data/meta-data/test2.json', 'r', encoding='utf-8') as f:
    for line in f:
        line = json.loads(line)
        examples[line['id']] = line['text']
enums = {}
with open('./results/enum/test2_pred_processed.json', 'r', encoding='utf-8') as f:
    for line in f:
        line = json.loads(line)
        enums[line['id']] = line['label']
def pkl2json(path, save_path):
    fw = open(save_path, 'w', encoding='utf-8')
    with open(path, 'rb') as f:

        content = pickle.load(f)
        #assert len(content) == len(examples)
        for index, c in enumerate(content):
            id = c[0]
            texts = examples[id]
           # id = ex['id']
            enum = enums.get(id)
            event_list = []
            for i in range(len(c[1])):
                if c[1][i]==1:

                    event_type = event_type_fields_list[i][0]
                    event_objs = c[2][i]
                    for event_obj in event_objs :
                        arguments = []
                        for j in range(len(event_type_fields_list[i][1])):
                            if event_obj[j] != None:
                                arg_id = event_obj[j]
                                pos = c[3].span_token_tup_list.index(arg_id)
                                sent_id, s, e = c[3].span_dranges_list[pos][0]
                                arg = texts[sent_id][s:e]
                                arg = arg.replace('®', ' ')
                                arguments.append({'role': event_type_fields_list[i][1][j], 'argument':arg })
                        #if arguments!=None:
                        if event_type == "公司上市":
                            arguments.append({'role': "环节", 'argument': enum})
                        event_list.append({'event_type': event_type ,'arguments':arguments})
            fw.write(json.dumps({'id':id, 'event_list': event_list},  ensure_ascii=False)+'\n')



if __name__ == "__main__":
    pkl2json('./Exps/doc2edag/Output/dee_eval.test2.gold_span.Doc2EDAG.32.pkl', './results/dee_eval.test2.gold_span.Doc2EDAG.32.json')