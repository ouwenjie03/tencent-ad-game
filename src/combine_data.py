# coding=utf-8
"""
    拼表和生成csv文件

"""

import numpy as np
import pandas as pd
import os
import sys

from args import *
from time import  time

# usage: python combine_data.py [chusai/fusai] [train/test1/test2]

assert sys.argv[1] in ['chusai', 'fusai']
assert sys.argv[2] in ['train', 'test1', 'test2']

is_train = 'train' in sys.argv[2]

ad_feature_path = '../data/{}/adFeature.csv'.format(sys.argv[1])
user_feature_path = '../data/{}/userFeature.data'.format(sys.argv[1])
raw_path = '../data/{}/{}.csv'.format(sys.argv[1], sys.argv[2])

ad_feature=pd.read_csv(ad_feature_path)

userFeature_data = []
user_feature = None
with open(user_feature_path, 'r') as f:
    for i, line in enumerate(f):
        line = line.strip().split('|')
        userFeature_dict = {}
        for each in line:
            each_list = each.split(' ')
            userFeature_dict[each_list[0]] = ' '.join(each_list[1:])
        userFeature_data.append(userFeature_dict)
        if i % 100000 == 0:
            print(i)
    user_feature = pd.DataFrame(userFeature_data)
user_feature['uid'] = user_feature['uid'].apply(int)

raw_data = pd.read_csv(raw_path)
if is_train:
    raw_data.loc[raw_data['label']==-1,'label']=0
else:
    raw_data['label']=-1

data=pd.merge(raw_data,ad_feature,on='aid',how='left')
data=pd.merge(data,user_feature,on='uid',how='left')
data=data.fillna('-1')

if sys.argv[1] == 'fusai':
    data.to_csv(args.root_data_path + '../data/combine_{}.csv'.format(sys.argv[2]), index=False)
else:
    data.to_csv(args.root_data_path + '../data/combine_{}_{}.csv'.format(sys.argv[1], sys.argv[2]), index=False)





