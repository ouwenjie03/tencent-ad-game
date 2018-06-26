#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 ouwj <ouwenjie>
#
# Distributed under terms of the MIT license.

"""

"""

import numpy as np
import sys, os
import pickle
from DataLoader import *
from args import *

def build_count2idx(data, labels, begin, end):
    total_count2idx = {"<pad>":0, -1:1, 0:2}
    pos_count2idx = {"<pad>":0, -1:1, 0:2}
    for i, d in enumerate(data):
        if i >= begin and i < end:
            continue
        if d not in total_count2idx:
            total_count2idx[d] = 2
        if d not in pos_count2idx:
            pos_count2idx[d] = 2
        total_count2idx[d] += 1
        if labels[i] == '1':
            pos_count2idx[d] += 1
    return total_count2idx, pos_count2idx

def build_test_data(test_data, pos_count2idx):
    test_res = []
    for d in test_data:
        if d not in pos_count2idx:
            test_res.append(1)
        else:
            test_res.append(pos_count2idx[d])
    return test_res


def count_pos_feature(train_data, labels, k=5, test1_data=None, test2_data=None):
    nums = len(train_data)
    last = nums
    interval = last // k
    parts = []
    for i in range(k):
        parts.append(i * interval)
    parts.append(last)
    count_train_data = train_data[0:last]
    count_labels = labels[0:last]

    train_res = []
    for i in range(k):
        sys.stderr.write("{}, part counting\n".format(i))
        sys.stderr.write("{}, {}\n".format(parts[i], parts[i+1]))
        tmp = []
        total_count2idx, pos_count2idx = build_count2idx(count_train_data, count_labels, parts[i], parts[i+1])
        for j in range(parts[i],parts[i+1]):
            d = train_data[j]
            if d not in pos_count2idx:
                tmp.append(1)
            else:
                tmp.append(pos_count2idx[d])
        train_res.extend(tmp)
    train_res = np.asarray(train_res)

    total_count2idx, pos_count2idx = build_count2idx(count_train_data, count_labels, 1, 0)

    test1_res = None
    if test1_data is not None:
        test1_res = build_test_data(test1_data, pos_count2idx)
        test1_res = np.asarray(test1_res)

    test2_res = None
    if test2_data is not None:
        test2_res = build_test_data(test2_data, pos_count2idx)
        test2_res = np.asarray(test2_res)

    max_idx = 0
    for key in pos_count2idx:
        if max_idx < pos_count2idx[key]:
            max_idx = pos_count2idx[key]
    max_idx += 1

    return train_res, test1_res, test2_res, max_idx

def save_bin(data, dl, name, fname):
   for p in range(dl.n_parts):
       sys.stderr.write("saving {} part {}\n".format(name, p))
       sp = p * dl.parts
       ep = (p+1) * dl.parts
       bin_file_path = os.path.join(args.root_data_path, 'bin_files/{}/{}_pos_{}.bin'.format(name, fname, p))
       data[sp:ep].tofile(bin_file_path, format="%d")


if __name__ == "__main__":
    train_dl = DataLoader(type_name="train_all", is_train=True)
    test1_dl = DataLoader(type_name="valid", is_train=False)
    test2_dl = DataLoader(type_name="test2", is_train=False)

    fnames = ['uid']
    fnames += ['uid|'+x for x in args.ad_static_features[1:]]

    tmp_fnames = []
    for fname in fnames:
        if fname not in tmp_fnames:
            tmp_fnames.append(fname)
        cf = fname.split('|')
        if len(cf) == 2:
            if cf[0] not in tmp_fnames:
                tmp_fnames.append(cf[0])
            if cf[1] not in tmp_fnames:
                tmp_fnames.append(cf[1])

    train_dl.load_data("../data/combine_train_all.csv", tmp_fnames)
    test1_dl.load_data("../data/combine_valid.csv", tmp_fnames)
    test2_dl.load_data("../data/combine_test2.csv", tmp_fnames)

    for fname in fnames:
        train_dl.combine_features(fname)
        test1_dl.combine_features(fname)
        test2_dl.combine_features(fname)

    for fname in fnames:
        if fname == 'label':
            continue
        train_data, test1_data, test2_data, max_idx = count_pos_feature(train_dl.id_data[fname], train_dl.id_data['label'], 5, test1_dl.id_data[fname], test2_dl.id_data[fname])
        #  train_data, test1_data, test2_data, max_idx = count_pos_feature(train_dl.id_data[fname], train_dl.id_data['label'], 5, test1_dl.id_data[fname], None)

        # save
        save_bin(train_data, train_dl, 'train_all', fname)
        save_bin(test1_data, test1_dl, 'valid', fname)
        save_bin(test2_data, test2_dl, 'test2', fname)

        # save_max_idx
        sys.stderr.write('max_idx: {}\n'.format(max_idx))
        max_idxs_file_path = args.root_data_path + "infos/max_idxs/{}_pos.pkl".format(fname)
        pickle.dump(max_idx, open(max_idxs_file_path, 'w'))

