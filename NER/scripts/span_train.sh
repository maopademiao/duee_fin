filename=role
# python -u ./ner.py \
# --architecture span \
# --bert_config_file /home/xuwd/data/bert-base-chinese \
# --train_file ./data/${filename}/train.txt \
# --dev_file ./data/${filename}/dev.txt \
# --test_file ./data/${filename}/test.txt \
# --output_dir ./output/${filename} \
# --max_len 512 \
# --tags_file ./data/${filename}/class.txt \
# --train_batch_size 10 \
# --learning_rate 5e-5 \
# --epoch 1 \
# --test_batch_size 5 \
# --tensorboard ./output/${filename}/logs \
# --dropout 0.5 \
# --seed 1

python -u ./ner.py \
--architecture span \
--bert_config_file /home/xuwd/data/bert-base-chinese \
--train_file ./data/${filename}/train.txt \
--dev_file ./data/${filename}/dev.txt \
--test_file ./data/${filename}/test2.txt \
--output_dir ./output/${filename} \
--max_len 512 \
--tags_file ./data/${filename}/class.txt \
--train_batch_size 10 \
--learning_rate 5e-5 \
--epoch 30 \
--test_batch_size 10 \
--tensorboard ./output/${filename}/logs \
--dropout 0.5 \
--seed 1


