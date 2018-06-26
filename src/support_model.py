#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 ouwj <ouwenjie>
#
# Distributed under terms of the MIT license.

"""
some utils for model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import pickle
import sys

from args import *

def weight_init(m):
    # 使用isinstance来判断m属于什么类型
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform(m.weight.data)
        #  nn.init.xavier_normal(m.weight.data)
        #  nn.init.normal(m.weight.data, 0.0, 0.0001)
        m.bias.data.fill_(0.)
        #  nn.init.normal(m.bias.data, 0.0, 0.001)
    elif isinstance(m, nn.Embedding):
        nn.init.xavier_uniform(m.weight.data)
        #  nn.init.xavier_normal(m.weight.data)
        #  nn.init.normal(m.weight.data, 0.0, 0.0001)

def make_mask(sen_lens, max_length, fill_val=True):
    """
    lengths_arr: one_dimension, list
    max_length: the max length of arr
    fill_val: the value filled in mask
    """

    batch_size = len(sen_lens)
    mask = torch.ByteTensor(batch_size, max_length).fill_(not fill_val)
    for b in range(batch_size):
        mask[b, 0:sen_lens[b]] = fill_val
    mask = torch.autograd.Variable(mask, requires_grad=False)
    return mask

def filter_same_features(fname1, fname2):
    in_ad_1 = fname1 in args.ad_static_features
    in_ad_2 = fname2 in args.ad_static_features
    in_user_1 = fname1 in (args.user_static_features + args.user_dynamic_features + args.len_static_features)
    in_user_2 = fname2 in (args.user_static_features + args.user_dynamic_features + args.len_static_features)
    in_ad = in_ad_1 and in_ad_2
    in_user = in_user_1 and in_user_2
    return in_ad or in_user
