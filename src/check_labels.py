#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 ouwj <ouwenjie>
#
# Distributed under terms of the MIT license.

"""

"""

import sys

valid_file = sys.argv[1]

label_cnt = [0, 0]
with open(valid_file, 'r') as fin:
    fin.readline()
    for line in fin:
        # find label
        parts = line.strip().split(',')
        label = parts[2]
        label_cnt[int(label)] += 1

print label_cnt
