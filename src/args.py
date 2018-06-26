#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 ouwj <ouwenjie>
#
# Distributed under terms of the MIT license.

"""
args
"""

class args:

    ignore_features = ['appIdInstall', 'appIdAction', 'marriageStatus', 'kw3', 'interest3', 'interest4', 'topic3']

    user_static_features = ['house', 'education', 'LBS', 'consumptionAbility', 'gender', 'age', 'carrier']
    ad_static_features = ['aid', 'advertiserId', 'campaignId', 'creativeId', 'creativeSize', 'adCategoryId','productId', 'productType']
    user_dynamic_features = ['interest1', 'interest2', 'interest5', 'kw1', 'kw2', 'topic1', 'topic2', 'ct', 'os']
    #  user_dynamic_features = ['interest1']
    #  len_static_features = [x+'_len' for x in user_dynamic_features] + ['uid']
    len_static_features = [x+'_len' for x in user_dynamic_features] + ['uid'] + ['uid_pos']
    #  len_static_features = [x+'_len' for x in user_dynamic_features] + ['uid'] + ['uid_pos'] + ['uid|{}_pos'.format(x) for x in ad_static_features[1:]]
    #  len_static_features = [x+'_len' for x in user_dynamic_features] + ['uid'] + ['uid_pos'] + ['uid|{}_pos'.format(x) for x in ['advertiserId', 'campaignId', 'adCategoryId']]
    #  len_static_features = ['interest1_len', 'interest2_len', 'interest5_len']
    #  len_static_features = [] # + ['uid']

    #  combine_features_1 = ['aid', 'productId', 'productType', 'advertiserId']
    #  combine_features_2 = ['LBS', 'gender', 'age', 'education', 'consumptionAbility']
    combine_features_1 = []
    combine_features_2 = []
    combine_features = [x+'|'+y for x in combine_features_1 for y in combine_features_2]

    static_features = ad_static_features + user_static_features + len_static_features
    #  static_features = ad_static_features + user_static_features
    dynamic_features = user_dynamic_features + combine_features

    all_features = static_features + dynamic_features

    root_train_data_path = '../data/'
    root_test_data_path = '../data/'
    root_data_path = '../data/'

    lr = 0.0015
    momentum = 0.9
    weight_decay = 0.00000
    dropout_rate = None
    batch_norm = True
    embedding_size = 16
    cut_threshold = 100

    use_cuda = True

    has_cr = False  # has conversion rate features

    epochs = 1
    batch_size = 1024

    n_train_parts = 11  # fusai + chusai
    n_test1_parts = 3
    n_test2_parts = 3
    n_valid_parts = 1
