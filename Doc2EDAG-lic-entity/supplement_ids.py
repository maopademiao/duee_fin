# 最后得到的结果缺失了id， 补全测试集id

import json

def supplement(path1, path2, save_path):
    meta_examples = []
    with open(path1, 'r') as f:
        for l in f:
            l = json.loads(l)
            meta_examples.append(l)
    print(len(meta_examples))
    final_results = []
    i = 0
    with open(path2, 'r') as f:
        for line in f:
            line = json.loads(line)
            while line['id'] != meta_examples[i]['id']:
                final_results.append({'id': meta_examples[i]['id'], 'event_list': []})
                #final_results.append(line)
                i=i+1

            final_results.append(line)
            i=i+1

    print(i)
    assert len(final_results) == len(meta_examples)
    fw = open(save_path, 'w', encoding='utf-8')
    for e in final_results:
        fw.write(json.dumps(e, ensure_ascii=False) + '\n')









if __name__ == "__main__":
    supplement('./Data/meta-data/test2.json', './results/test2.all_cpt.json', './results/test2.all_cpt_all_ids.json')