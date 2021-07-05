import json

def fusion(path1, path2, save_path):
    examples1, examples2 = [], []
    examples1_id, examples2_id = [], []
    with open(path1, 'r') as f:
        for line in f:
            event_list = {}
            line = json.loads(line)
            for event in line['event_list']:
                new_arguments = {}
                for role in event['arguments']:
                    new_arguments[role['role']] = role['argument']
                event_list[event['event_type']] = new_arguments
            examples1.append(event_list )
            examples1_id.append(line['id'])
    with open(path2, 'r') as f:
        for line in f:
            event_list = {}
            line = json.loads(line)
            for event in line['event_list']:
                new_arguments = {}
                for role in event['arguments']:
                    new_arguments[role['role']] = role['argument']
                event_list[event['event_type']] = new_arguments
            examples2.append(event_list)
            examples2_id.append(line['id'])
    assert len(examples1) == len(examples2)
    final_examples = []
    for i in range(len(examples1)):
        ex1 = examples1[i]
        ex2 = examples2[i]
        assert examples1_id [i] == examples2_id [i]
        event_type_list1 = ex1.keys()
        event_type_list2 = ex2.keys()
        for event in event_type_list1:
            if event in event_type_list2:
                arguments1 = ex1[event]
                arguments2 = ex2[event]
                key = arguments1.keys()
                for r, arg in arguments2.items():
                    if r not in key:
                        arguments1[r] = arg
                ex1[event] = arguments1
        for event in event_type_list2:
            if event not in event_type_list1:
                ex1[event] = ex2[event]

        final_examples.append(ex1)

    fw = open(save_path, 'w', encoding='utf-8')
    for id, e in zip(examples1_id, final_examples):
        event_list = []
        for event_type, argu in e.items():
            arguments = []
            for role, argument in argu.items():
                arguments.append({'role':role, 'argument': argument})
            event = {'event_type': event_type , 'arguments':arguments}
            event_list.append(event)

        fw.write(json.dumps({'id': id, 'event_list': event_list}, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    fusion('./results/dee_eval.test2.pred_span.Doc2EDAG.32-46-26.json', './results/dee_eval.test2.pred_span.Doc2EDAG.34-29.json', './results/test2.all_cpt.json')