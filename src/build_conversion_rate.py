#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 ouwj <ouwenjie>
#
# Distributed under terms of the MIT license.

"""

"""

import argparse
import sys, os

from DataLoader import *
from args import *

parser = argparse.ArgumentParser()
parser.add_argument("--type_name", type=str, default="train_shuf", help="[train_shuf, test1, test2]")
parser_args = parser.parse_args()
assert parser_args.type_name in ["train", "train_shuf", "test1", "test2", "valid", "train_all"]

is_train = 'train' in parser_args.type_name
if is_train:
    args.root_data_path = args.root_train_data_path
dataLoader = DataLoader(type_name=parser_args.type_name, is_train=is_train)

n_parts = args.n_train_parts
if parser_args.type_name == 'test1':
    n_parts = args.n_test1_parts
elif parser_args.type_name == 'test2':
    n_parts = args.n_test2_parts
elif parser_args.type_name == 'valid':
    n_parts = args.n_valid_parts


#  now_features = args.user_static_features
now_features = args.ad_static_features + args.user_static_features + args.user_dynamic_features
#  now_features = args.dynamic_features
file_path = os.path.join(args.root_data_path, "combine_{}.csv".format(parser_args.type_name))
for fname in now_features:
    sys.stderr.write('saving {} conversion rate\n'.format(fname))
    dataLoader.build_conversion_rate(fname)
    #  for p in range(n_parts):
    #      dataLoader.save_conversion_rate(fname, p)
