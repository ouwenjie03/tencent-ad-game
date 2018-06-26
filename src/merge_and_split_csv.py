#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 ouwj <ouwenjie>
#
# Distributed under terms of the MIT license.

"""

"""

import sys, random

is_shuf = True
split_valid = 0.02

merge_data = []

csvs = sys.argv[1:]
for csv in csvs:
    with open(csv, 'r') as fin:
        head = fin.readline().strip()
        for line in fin:
            merge_data.append(line.strip())

# random shuffle
total_len = len(merge_data)
split = min(int(total_len * split_valid), 1000000)
idxs = range(total_len)
if is_shuf:
    random.shuffle(idxs)

if split_valid > 0:
    train_idxs = idxs[0:(total_len-split)]
    valid_idxs = idxs[(total_len-split):]
else:
    train_idxs = idxs
    valid_idxs = []

train_file_name = "../data/combine_train_all.csv"
with open(train_file_name, 'w') as fout:
    fout.write(head+'\n')
    for i in train_idxs:
        fout.write(merge_data[i]+'\n')

if split_valid > 0:
    valid_file_name = "../data/combine_valid.csv"
    with open(valid_file_name, 'w') as fout:
        fout.write(head+'\n')
        for i in valid_idxs:
            fout.write(merge_data[i]+'\n')

