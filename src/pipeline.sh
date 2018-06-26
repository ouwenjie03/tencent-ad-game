#! /bin/sh
#
# pipeline.sh
# Copyright (C) 2018 ouwj <ouwenjie>
#
# Distributed under terms of the MIT license.
#

# mkdir useful dir
mkdir -p ../data/bin_files
mkdir -p ../data/bin_files/train_all
mkdir -p ../data/bin_files/valid
mkdir -p ../data/bin_files/test2
mkdir -p ../data/infos
mkdir -p ../data/infos/max_idxs
mkdir -p ../data/infos/conversion_infos
mkdir -p ../data/infos/train_all/max_lens
mkdir -p ../data/infos/valid/max_lens
mkdir -p ../data/infos/test2/max_lens
mkdir -p ../data/feature2idx
mkdir -p ../models
mkdir -p ../result
mkdir -p logs

# combine three data file
python combine_data.py chusai train
python combine_data.py fusai train
python combine_data.py fusai test2

# merge and split dataset
python merge_and_split_csv.py ../data/combine_train.csv ../data/combine_chusai_train.csv

# build uid2idx
python build_uid2idx.py

# build index data
# type_name = [train_all, valid, test1, test2], but first must be train_all to build index dict
python DataLoader.py --type_name train_all
python DataLoader.py --type_name valid
python DataLoader.py --type_name test2

# build other max_idxs
python build_len_max_idx.py

# build pos count feature in [train_all, valid and test2]
python build_pos_feature.py

# count and save 
python build_conversion_rate.py --type_name train_all

# train model
CUDA_VISIBLE_DEVICES=0 python main.py --type_name train_all --is_valid 1 --model_name NFFM_concat

# predict model
CUDA_VISIBLE_DEVICES=0 python main.py --type_name test2 --is_valid 0 --model_name NFFM_concat --model_path ../models/NFFM_concat.pkl > ../result/NFFM_concat.csv

# make submission
python make_submission.py ../result/NFFM_concat.csv ../result/submission.csv



