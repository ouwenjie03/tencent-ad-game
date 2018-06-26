#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 ouwj <ouwenjie>
#
# Distributed under terms of the MIT license.

"""
train and test
"""

import torch
import torch.nn as nn

from sklearn.metrics import roc_auc_score

import numpy as np
import sys
import time, datetime
import pickle
import random

from DataLoader import *
from NFFM import *
from NFFM_concat import *
from NFFM_concat_dot import *
from NFFM_concat_triple import *
from args import *
from support_model import *

def fit(model, data_loader, st_fnames, dy_fnames, parts, model_name, valid_data_loader=None):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(1, args.epochs+1):
        stime = time.time()
        cnt = 0
        total_loss = 0.0
        random_parts = parts
        random.shuffle(random_parts)
        for part in random_parts:
            data_loader.load_bin(st_fnames+dy_fnames, part)
            if args.has_cr:
                data_loader.load_conversion_rate_from_bin(st_fnames+dy_fnames, part)
            data_loader.reset()
            data_loader.random_shuffle(st_fnames+dy_fnames)
            while True:
                data = data_loader.next_batch([st_fnames, dy_fnames], args.batch_size)
                if data is None:
                    break
                
                st_ids = data[0]
                dy_ids = data[1]
                dy_lens = data[2]
                labels = data[3]

                #  sys.stderr.write('{}\n'.format(st_ids))
                #  sys.stderr.write('{}\n'.format(dy_ids))

                conversion_rates = data[4]

                optimizer.zero_grad()
                scores = model(st_ids, dy_ids, dy_lens, conversion_rates)
                loss = model.get_loss(scores, labels)
                total_loss += loss.data[0]
                loss.backward()
                optimizer.step()
                cnt += 1

                if cnt % 100 == 0:
                    etime = time.time()
                    sys.stderr.write("epoch:{} | batch:{} | avg loss:{} | cost:{}s\n".format(epoch, cnt, total_loss/100, etime-stime))
                    total_loss = 0.0
                    stime = etime

            if valid_data_loader is not None:
                predict(model, valid_data_loader, st_fnames, dy_fnames, range(args.n_valid_parts), is_valid=True)

            args.lr = args.lr * 0.9
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        #  now_time_str = datetime.datetime.now().strftime("%Y-%m-%d~%H:%M:%S")
        #  model_name = '../models/{}_{}_epoch_{}.pkl'.format(model_name, now_time_str, epoch)
        model_name = '../models/{}.pkl'.format(model_name)
        sys.stderr.write("saving model in {}...\n".format(model_name))
        model.save(model_name)


def predict(model, data_loader, st_fnames, dy_fnames, parts, is_valid=False):
    model.eval()
    total_loss = 0.0
    cnt = 1
    total_y_true = []
    total_y_predict = []
    for part in parts:
        data_loader.load_bin(st_fnames+dy_fnames, part)
        if args.has_cr:
            data_loader.load_conversion_rate_from_bin(st_fnames+dy_fnames, part)
        data_loader.reset()
        while True:
            data = data_loader.next_batch([st_fnames, dy_fnames], args.batch_size)
            if data is None:
                break

            st_ids = data[0]
            dy_ids = data[1]
            dy_lens = data[2]

            conversion_rates = data[4]

            scores = model(st_ids, dy_ids, dy_lens, conversion_rates)
            probs = torch.sigmoid(scores)

            if not is_valid:
                for p in probs.data:
                    print("%.6f" % p)
            else:
                labels = data[3]
                total_y_true += [x for x in labels]
                total_y_predict += [x for x in scores.data]
                loss = model.get_loss(scores, labels)
                total_loss += loss.data[0] * len(labels)
                cnt += len(labels)
    if is_valid:
        total_auc = roc_auc_score(np.asarray(total_y_true), np.asarray(total_y_predict))
        sys.stderr.write("valid loss:{} | valid auc:{}\n".format(total_loss/cnt, total_auc))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--type_name", type=str, default="train_shuf", help="[train_shuf, test1, test2]")
    parser.add_argument("--is_valid", type=int, default=0, help="args for train: 1 for valid, 0 for not valid")
    parser.add_argument("--model_name", type=str, default="", help="model_name")
    parser.add_argument("--model_path", type=str, default="", help="model_path for test load")
    parser_args = parser.parse_args()

    assert parser_args.type_name in ["train", "train_shuf", "test1", "test2", "train_all", "valid"]

    is_train = 'train' in parser_args.type_name
    if is_train:
        args.root_data_path = args.root_train_data_path
    data_loader = DataLoader(type_name=parser_args.type_name, is_train=is_train, has_cr=args.has_cr)

    st_fnames = args.static_features
    st_max_idxs = data_loader.get_max_idxs(st_fnames)
    dy_fnames = args.dynamic_features
    dy_max_idxs = data_loader.get_max_idxs(dy_fnames)
    
    # for cut
    data_loader.load_counts(st_fnames+dy_fnames)

    # for valid
    is_valid = parser_args.is_valid == 1    
    valid_data_loader = None
    if is_valid:
        valid_data_loader = DataLoader(type_name="valid", is_train=is_train, has_cr=args.has_cr)
        valid_data_loader.load_max_idxs(st_fnames+dy_fnames)
        valid_data_loader.load_counts(st_fnames+dy_fnames)



    sys.stderr.write("st_features:{}\n".format(st_fnames))
    sys.stderr.write("dy_features:{}\n".format(dy_fnames))

    sys.stderr.write("building model...\n")
    if parser_args.model_name == 'NFFM':
        model = DyNFFM(
            [st_fnames, dy_fnames],
            [st_max_idxs, dy_max_idxs],
            embedding_size=args.embedding_size,
            dropout_rate = args.dropout_rate,
            batch_norm=args.batch_norm,
            use_cuda=args.use_cuda
            )
    elif parser_args.model_name == 'NFFM_concat':
        model = DyNFFM_concat(
            [st_fnames, dy_fnames],
            [st_max_idxs, dy_max_idxs],
            embedding_size=args.embedding_size,
            dropout_rate = args.dropout_rate,
            batch_norm=args.batch_norm,
            use_cuda=args.use_cuda
            )
    elif parser_args.model_name == 'NFFM_concat_triple':
        model = DyNFFM_concat_triple(
            [st_fnames, dy_fnames],
            [st_max_idxs, dy_max_idxs],
            embedding_size=args.embedding_size,
            dropout_rate = args.dropout_rate,
            batch_norm=args.batch_norm,
            use_cuda=args.use_cuda
            )
    elif parser_args.model_name == 'NFFM_concat_dot':
        model = DyNFFM_concat_dot(
            [st_fnames, dy_fnames],
            [st_max_idxs, dy_max_idxs],
            embedding_size=args.embedding_size,
            dropout_rate = args.dropout_rate,
            batch_norm=args.batch_norm,
            use_cuda=args.use_cuda
            )
    else:
        model = None

    model.apply(weight_init)
    if args.use_cuda:
        model.cuda()

    if "train" in parser_args.type_name:

        fit(model, data_loader, st_fnames, dy_fnames, range(args.n_train_parts), parser_args.model_name, valid_data_loader)

    elif "test" in parser_args.type_name:
        sys.stderr.write("loading model in {}...\n".format(parser_args.model_path))
        model.load(parser_args.model_path)
        n_test_parts = args.n_test1_parts
        if parser_args.type_name == 'test2':
            n_test_parts = args.n_test2_parts
        predict(model, data_loader, st_fnames, dy_fnames, range(n_test_parts), is_valid=False)
        

        

