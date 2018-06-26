#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 ouwj <ouwenjie>
#
# Distributed under terms of the MIT license.

"""
avg submission
"""

import sys

total_cnt = len(sys.argv[1:])
total_res = []

cnt = 0
for f in sys.argv[1:]:
    with open(f, 'r') as fin:
        idx = 0
        for line in fin:
            line = line.strip()
            if cnt == 0:
                total_res.append(float(line))
            else:
                total_res[idx] += float(line)
            idx += 1
    cnt += 1

for res in total_res:
    print('%.6f' % (res/total_cnt))

