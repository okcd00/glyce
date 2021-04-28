# -*- coding: utf-8 -*-
repo_path=/home/chendian/glyce/glyce
data_sign=msra_ner


CUDA_VISIBLE_DEVICES=0 python ${repo_path}/bin/run_lattice_lstm.py \
--name ${data_sign} \
--status train \
