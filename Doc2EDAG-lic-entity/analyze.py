import json


#对每个example，选一个概率最大的enum
def enum_process(path, save_path):
    examples = []
    pre_id = ""
    label = ""
    prob = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            cur_id = line['id']
            cur_label = line['pred']['label']
            if cur_id==pre_id:
                if  line['pred']['probs'][cur_label ]>prob:
                    label = cur_label
            else:
                examples.append({'id':pre_id, 'label':label})
                pre_id = cur_id
                label = line['pred']['label']
                prob = line['pred']['probs'][cur_label ]
    fw = open(save_path, 'w', encoding='utf-8')
    max_enum = 0
    for exa in examples[1:]:
        #max_enum = max(max_enum, len(exa['labels']))
        fw.write(json.dumps(exa, ensure_ascii=False)+'\n')
    fw.close()
    print(max_enum)




if __name__ == "__main__":
    enum_process('/home/zy/Doc2EDAG-lic-entity/results/enum/test2_pred.json', '/home/zy/Doc2EDAG-lic-entity/results/enum/test2_pred_processed.json')