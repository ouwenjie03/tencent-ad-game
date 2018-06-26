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
import random

file_name = sys.argv[1]
with open(file_name, 'r') as fin:
    head = fin.readline()
    data = fin.readlines()
    idxs = range(len(data))
    random.shuffle(idxs)

    print(head)
    for i in idxs:
        print data[i].strip()
