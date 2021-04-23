# -*- coding: utf-8 -*-
repo_path=/home/chendian/glyce/glyce

data_sign=msra_ner
data_dir=/home/chendian/download/msra_ner
output_dir=/home/chendian/glyce/output/

config_path=/home/chendian/glyce/glyce/configs/glyce_bert.json
bert_model=/home/chendian/download/ShannonBert

task_name=ner
max_seq_len=500
train_batch=32
dev_batch=32
test_batch=32
learning_rate=2e-5
num_train_epochs=10
warmup=0.1
local_rank=-1
seed=3306
checkpoint=250


CUDA_VISIBLE_DEVICES=0 python ${repo_path}/bin/run_bert_glyce_tagger.py \
--data_sign ${data_sign} \
--config_path ${config_path} \
--data_dir ${data_dir} \
--bert_model ${bert_model} \
--task_name ${task_name} \
--max_seq_length ${max_seq_len} \
--do_train \
--do_eval \
--train_batch_size ${train_batch} \
--dev_batch_size ${dev_batch} \
--test_batch_size ${test_batch} \
--learning_rate ${learning_rate} \
--num_train_epochs ${num_train_epochs} \
--checkpoint ${checkpoint} \
--warmup_proportion ${warmup} \
--output_dir ${output_dir} 
