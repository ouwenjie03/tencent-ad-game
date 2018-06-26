#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 ouwj <ouwenjie>
#
# Distributed under terms of the MIT license.

"""

"""

from args import *
import numpy as np
import sys, os
import pickle

args.root_data_path = args.root_train_data_path
max_idxs_file_path = args.root_data_path + "infos/max_idxs/{}.pkl"


def save_max_idx(fname, n_parts):
    max_idx = 0
    for name, n_parts in zip(['train_all'], [args.n_train_parts]):
        ###
        for p in range(n_parts):
            sys.stderr.write('loading {} part {}...\n'.format(fname, p))
            bin_file_path = os.path.join(args.root_data_path, 'bin_files/{}/{}_{}.bin'.format(name, fname, p))
            len_data = np.fromfile(bin_file_path, dtype=int)
            tmp = np.max(len_data)
            if tmp > max_idx:
                max_idx = tmp
        #  pickle.dump(max_idx, open(max_idxs_file_path.format(fname+'_len'), 'w'))
    pickle.dump(max_idx+1, open(max_idxs_file_path.format(fname), 'w'))

all_fnames = ['{}_len'.format(x) for x in args.user_dynamic_features]
for fname in all_fnames:
    save_max_idx(fname, args.n_train_parts)
