#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 ouwj <ouwenjie>
#
# Distributed under terms of the MIT license.

import sys


f1 = open('../data/test2.csv')
f2 = open(sys.argv[1])
f = open(sys.argv[2],'wb')

f.write('aid,uid,score\n')
f1.readline()
for line in f1:
    line = line.strip() +','+ f2.readline()
    f.write(line)
