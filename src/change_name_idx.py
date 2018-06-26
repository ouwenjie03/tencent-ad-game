#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 ouwj <ouwenjie>
#
# Distributed under terms of the MIT license.

"""

"""

import sys, os

dir_path = sys.argv[1]
names = os.listdir(dir_path)
for name in names:
    pre_name, pos_name = name.split('.')
    parts = pre_name.split('_')
    parts[-1] = str(int(parts[-1]) + 10)
    new_name = '_'.join(parts) + '.' + pos_name
    #  new_name = name.replace('bin', '.bin')
    print(new_name)

    os.system('mv {} {}'.format(dir_path+'/'+name, dir_path+'/'+new_name))
