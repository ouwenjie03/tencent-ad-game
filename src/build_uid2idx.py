#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 ouwj <ouwenjie>
#
# Distributed under terms of the MIT license.

"""

"""

import os, sys
import pickle
import numpy as np

from args import *

file_format = os.path.join(args.root_data_path, "combine_{}.csv")
#  files = ['train_all', 'test1', 'test2', 'valid']
files = ['train_all']

uid2cnt = {'<pad>':0, '-1':1, '<unk>':2}
for f in files:
    cnt = 1
    file_name = file_format.format(f)
    with open(file_name, 'r') as fin:
        fnames = fin.readline().strip().split(',')
        # find uid
        idx = 0
        for i in range(len(fnames)):
            if fnames[i] == 'uid':
                idx = i
                break

        for line in fin:
            datas = line.strip().split(',')
            uid = datas[idx]
            if uid not in uid2cnt:
                uid2cnt[uid] = 2
            uid2cnt[uid] += 1

            if cnt % 1000000 == 0:
                sys.stderr.write('loading {} part {}...\n'.format(f, cnt))
            cnt += 1

uid2idx_file = os.path.join(args.root_data_path, "feature2idx/uid2idx.pkl")
#  uid2cnt = pickle.load(open(uid2idx_file, 'rb'))

max_idx = 0
for uid in uid2cnt:
    #  uid2cnt[uid] = int(np.log(uid2cnt[uid] ** 2 + 1))
    if uid2cnt[uid] > max_idx:
        max_idx = uid2cnt[uid]
sys.stderr.write('max_idx:{}\n'.format(max_idx))
uid2cnt['max_idx'] = max_idx + 1

# save 
pickle.dump(uid2cnt, open(uid2idx_file, 'wb'))

