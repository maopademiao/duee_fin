import os
import argparse
import json


parser = argparse.ArgumentParser(description="Make data")
parser.add_argument("--type", default='role', type=str,
                        help="[trigger,role]")

args = parser.parse_args()


def make_class(train_file, class_file):
    class_list = []
    with open(train_file,'r',encoding='utf-8') as f:
        num = 0
        for lines in f:
            if num==0:
                num+=1
                continue
            # print(lines)
            line = lines.strip('\n').split('\t')
            # print(line[0])
            # print(line[1].split('\002'))
            labels = line[1].split('\002')
            for i in range(0,len(labels)):
                if labels[i]=='O':
                    continue
                else:
                    newlabel = labels[i].split('-')[1]
                    # print(newlabel)
                    if newlabel not in class_list:
                        class_list.append(newlabel)

            num += 1
            # if num==2:break
    with open(class_file, 'w', encoding='utf-8') as f:
        for key in class_list:
            f.write(key+'\n')

def make_file(train_path,write_path):
    wf = open(write_path,'w',encoding='utf-8')
    with open(train_path,'r',encoding='utf-8') as f:
        num = 0
        for lines in f:
            if num==0:
                num+=1
                continue
            line = lines.strip('\n').split('\t')
            # print(line[0])
            texts = line[0].split('\002')
            labels = line[1].split('\002')
            # print(texts)
            # print(labels)
            assert len(texts)==len(labels)
            for i in range(len(texts)):
                text = texts[i]
                label = labels[i]
                wf.write(text+'\t'+label+'\n')
            wf.write('\n')
            # num+=1
            # if num==2:break
    wf.close()

def make_test(test_path,write_path):
    wf = open(write_path,'w',encoding='utf-8')
    with open(test_path,'r',encoding='utf-8') as f:
        for lines in f:
            line = json.loads(lines)
            # print(line)
            text = line.get('text')
            # print(text)
            for i in range(0,len(text)):
                wf.write(text[i]+'\t'+'O'+'\n')
            wf.write('\n')
    wf.close()

def make_trigger():
    train_path = '/home/xuwd/projects/DuEE-pytorch/data/DuEE-Fin/trigger/train.tsv'
    dev_path = '/home/xuwd/projects/DuEE-pytorch/data/DuEE-Fin/trigger/dev.tsv'
    test_path  = '/home/xuwd/projects/DuEE-pytorch/data/DuEE-Fin/sentence/test2.json'

    write_train_path='./data/trigger/train.txt'
    write_dev_path='./data/trigger/dev.txt'
    write_test_path='./data/trigger/test2.txt'

    write_class_path='./data/trigger/class.txt'
    # make_class(train_path,write_class_path)
    # make_file(train_path,write_train_path)
    # make_file(dev_path, write_dev_path)
    make_test(test_path,write_test_path)

def make_role():
    train_path = '/home/xuwd/projects/DuEE-pytorch/data/DuEE-Fin/role/train.tsv'
    dev_path = '/home/xuwd/projects/DuEE-pytorch/data/DuEE-Fin/role/dev.tsv'
    test_path = '/home/xuwd/projects/DuEE-pytorch/data/DuEE-Fin/sentence/test2.json'

    write_train_path = './data/role/train.txt'
    write_dev_path = './data/role/dev.txt'
    write_test_path = './data/role/test2.txt'

    write_class_path = './data/role/class.txt'
    # make_class(train_path,write_class_path)
    # make_file(train_path,write_train_path)
    # make_file(dev_path, write_dev_path)
    make_test(test_path, write_test_path)


def main():
    if args.type=='trigger':
        make_trigger()
    else:
        make_role()

if __name__=='__main__':
    main()


