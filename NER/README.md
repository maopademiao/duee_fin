# Named Entity Recognition

命名实体识别（Pytorch），支持BERT-SPAN、BERT-CRF、BERT-SoftMax等模型。

## 文件目录

| 文件 | 描述 |
| :----: | :----: |
| _bert-base-chinese_ | BERT预训练模型文件（pytorch） |
| _data_ | 数据（train, dev, test, tags） |
| _ner.py_ | 多领域命名实体识别（pytorch） |

## 数据格式说明

数据文件（*train*，*dev*，*test*）采用BIO的标注格式，其中每行为一个字符和一个标签（中间以\t分开），空行表示一句话结束。详细可以看*data*中的样例。

SPAN方法使用为实体类别（如*class.txt*），如PER（数据文件中被标记为*B-PER*，*I-PER*，非实体为*O*）。

CRF和SoftMax方法使用数据中全部的标签类别（如*tag.txt*）。

## 环境参数

python        --3.6

pytorch-crf   --0.7.2 （CRF方法需要）

torch         --1.4.0 

torchvision   --0.5.0

tensorboard   --2.3.0 

tensorboardX  --2.1

transformers  --3.1.0

tqdm          --4.49.0

注：最新版本安装transformers时，将sentencepiece降到0.1.91版本，否则可能报错。

## 可选参数

| 参数 | 描述 | 解释 |
| :---- | :---- | :---- |
|-h, --help | show this help message and exit | |
|--train_file | TRAIN_FILE The training file path. | 训练数据 |
|--dev_file DEV_FILE |  The development file path. | 验证数据 |
|--test_file TEST_FILE | The testing file path. | 测试数据 |
|--tags_file TAGS_FILE | The tags file path. | 标签数据 |
|--output_dir OUTPUT_DIR | The output folder path. | 输出文件夹 |
|--model MODEL | The model path. | 验证和测试的模型路径 |
|--architecture {span,crf} | The model architecture of neural network and what decoding method is adopted. | 模型可选{span，crf} |
|--train_batch_size TRAIN_BATCH_SIZE | The number of sentences contained in a batch during training. | 训练的一批句子数 |
|--test_batch_size TEST_BATCH_SIZE |The number of sentences contained in a batch during testing. |验证或测试的一批句子数 |
|--epochs EPOCHS  | Total number of training epochs to perform. | 训练最大轮数 |
|--learning_rate LEARNING_RATE | The initial learning rate for Adam. | 学习率 |
|--crf_lr CRF_LR | The initial learning rate of CRF layer. | CRF层的学习率 |
|--dropout DROPOUT | What percentage of neurons are discarded in the fully connected layers (0 ~ 1). | 全连接层Dropout丢失率 |
|--max_len MAX_LEN | The Maximum length of a sentence. | 句子最大长度（如果实际句子过长则按照split集切分） |
|--keep_last_n_checkpoints KEEP_LAST_N_CHECKPOINTS | Keep the last n checkpoints. | 保留最后的几轮模型 | 
|--warmup_proportion WARMUP_PROPORTION |Proportion of training to perform linear learning rate warmup for. | warmup |
|--split SPLIT | Characters that segments a sentence. | 句子可以切分的字符（如标点） |
|--tensorboard_dir TENSORBOARD_DIR | The data address of the tensorboard. | Tensorboard路径 |
|--bert_config_file BERT_CONFIG_FILE | The config json file corresponding to the pre-trained BERT model. This specifies the model architecture. | BERT预训练模型 |
|--cpu  | Whether to use CPU, if not and CUDA is avaliable can use CPU. | 如果使用CPU |
|--seed SEED | random seed for initialization. | 随机种子 |

## 可选参数特殊说明

--train_file --dev_file --test_file 分别同时代表运行方式
>+ 只使用 __--train_file__ 则只训练到固定轮数，保存为最后的模型 *checkpoint-last.kpl*
>+ 若使用 __--train_file__ 和 __--dev_file__ 则会额外域保存在开发集上的最高分数的模型 *checkpoint-best.kpl*
>+ __--test-file__ 则为测试方式如存在 *checkpoint-best.kpl* 则使用该模型，否则使用 *checkpoint-last.kpl*

### crf方法

--crf_lr 有效，对CRF层设置不同的学习率

### -model作用

如指定则在验证测试的时候使用该模型，否则使用验证集上的最高分时模型（*checkpoint-best.pkl*）,若无验证集则使用模型训练最后的模型（*checkpoint-last.pkl*）

## 脚本样例

SPAN方法 ./scripts/span_train.sh

CRF方法 ./scripts/crf_train.sh

SotfMax方法 ./scripts/sotfmax_train.sh
