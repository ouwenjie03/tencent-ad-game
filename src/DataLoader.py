#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 ouwj <ouwenjie>
#
# Distributed under terms of the MIT license.

"""
data loader
"""

import pickle
import sys, os
import numpy as np
import time
import pickle

from args import *


class DataLoader:
    def __init__(self, type_name, is_train=True, has_cr=False, parts=1024*5000):
        self.type_name = type_name
        self.is_train=is_train
        self.has_cr = has_cr
        self.parts=parts

        self.max_lens_file_path = args.root_data_path + "/infos/{}/max_lens/{}.pkl"
        self.max_idxs_file_path = args.root_data_path + "infos/max_idxs/{}.pkl"

        # for preload
        self.id_data = {}
        self.len_data = {}
        self.feature2idx = {}
        self.max_idxs = {}

        self.label_counts = [0, 0]
        self.fname_counts = [{}, {}]

        self.feature_counts = {}

        # info
        self.n_all_data = -1
        self.next_idx = 0

        # for batch load
        self.ids = {}
        self.id_lens = {}
        self.max_lens = {}
        self.labels = []
        self.conversion_rates = {}

    # for preload
    def clear(self):
        del self.id_data, self.len_data, self.feature2idx, self.max_idxs, self.ids, self.id_lens, self.max_lens, self.labels
        self.id_data = {}
        self.len_data = {}
        self.feature2idx = {}
        self.max_idxs = {}

        self.n_all_data = -1
        self.next_idx = 0

        self.ids = {}
        self.id_lens = {}
        self.max_lens = {}
        self.labels = []


    def load_data(self, file_name, fnames):
        fnames.append("label")
        with open(file_name, 'r') as fin:
            features = fin.readline().strip().split(',')
            num = 0
            for line in fin:
                datas = line.strip().split(',')
                for i, d in enumerate(datas):
                    if features[i] in args.ignore_features or features[i] not in fnames:
                        continue
                    if features[i] not in self.id_data:
                        self.id_data[features[i]] = []
                    self.id_data[features[i]].append(d)
                num += 1
                if num % 1000000 == 0:
                    sys.stderr.write("loading {} data...\n".format(num))
                #  if num >= self.part_num:
                #      break
            self.n_all_data = num
            sys.stderr.write("total {} data...\n".format(num))

        # for save
        self.n_parts = self.n_all_data // self.parts
        if self.n_all_data % self.parts != 0:
            self.n_parts += 1

    def count_features(self, fname):
        if fname in args.dynamic_features:
            sys.stderr.write("counting {}...\n".format(fname))
            for data in self.id_data[fname]:
                if fname not in self.len_data:
                    self.len_data[fname] = []
                self.len_data[fname].append(len(data.split(' ')))
            self.len_data[fname] = np.asarray(self.len_data[fname])
        else:
            sys.stderr.write("Warning: {} is static feature!\n".format(fname))

    def build_feature2idx(self, fname):
        sys.stderr.write("building {}2idx dict...\n".format(fname))
        fname2idx_path = os.path.join(args.root_data_path, "feature2idx/{}2idx.pkl".format(fname))
        if os.path.exists(fname2idx_path):
            fname2idx = pickle.load(open(fname2idx_path, 'rb'))
        else:
            fname2idx = {"<pad>":0, '-1':1, '<unk>':2}
            now_idx = len(fname2idx)
            for data in self.id_data[fname]:
                for d in data.split(' '):
                    if d not in fname2idx:
                        fname2idx[d] = now_idx
                        now_idx += 1
            fname2idx["max_idx"] = now_idx
            pickle.dump(fname2idx, open(fname2idx_path, 'wb'))
        self.feature2idx[fname] = fname2idx

        # save max_idxs
        self.max_idxs[fname] = self.feature2idx[fname]["max_idx"]

    def combine_features(self, fname):
        cfs = fname.split('|')
        if len(cfs) != 2:
            return
        fname1 = cfs[0]
        fname2 = cfs[1]
        self.id_data[fname] = []
        sys.stderr.write("combining {} | {}...\n".format(fname1, fname2))
        for data1, data2 in zip(self.id_data[fname1], self.id_data[fname2]):
            combine_data = []
            for d1 in data1.split(' '):
                for d2 in data2.split(' '):
                    combine_data.append(d1+'|'+d2)
            self.id_data[fname].append(' '.join(combine_data))

    def get_max_idxs(self, fnames):
        # load max_idxs
        max_idxs = []
        for fname in fnames:
            max_idx = pickle.load(open(self.max_idxs_file_path.format(fname), 'r'))
            self.max_idxs[fname] = max_idx
            max_idxs.append(max_idx)
        return max_idxs
    
    def load_max_idxs(self, fnames):
        for fname in fnames:
            max_idx = pickle.load(open(self.max_idxs_file_path.format(fname), 'r'))
            self.max_idxs[fname] = max_idx

    def load_max_lens(self, fnames, part):
        for fname in fnames:
            if fname in args.static_features:
                continue
            max_len = pickle.load(open(self.max_lens_file_path.format(self.type_name, fname), 'r'))
            self.max_lens[fname] = max_len

    def save_max_lens(self, fname):
        if fname in args.static_features:
            return
        pickle.dump(self.max_lens[fname], open(self.max_lens_file_path.format(self.type_name, fname), 'w'))

    def save_max_idxs(self, fnames):
        for fname in fnames:
            pickle.dump(self.max_idxs[fname], open(self.max_idxs_file_path.format(fname), 'w'))


    # preload api
    def prepare_for_final_data(self, fnames):
        """
        preload data pipeline api
        """
        tmp_fnames = []
        for fname in fnames:
            if fname not in tmp_fnames:
                tmp_fnames.append(fname)
            cf = fname.split('|')
            if len(cf) == 2:
                if cf[0] not in tmp_fnames:
                    tmp_fnames.append(cf[0])
                if cf[1] not in tmp_fnames:
                    tmp_fnames.append(cf[1])

        # load csv
        stime = time.time()
        self.load_data(os.path.join(args.root_data_path, "combine_{}.csv".format(self.type_name)), tmp_fnames)
        etime = time.time()
        sys.stderr.write("load_data cost {} s...\n".format(etime-stime))

        # combine
        stime = time.time()
        for cf in fnames:
            self.combine_features(cf)
        etime = time.time()
        sys.stderr.write("combine_features cost {} s...\n".format(etime-stime))

        # count dynamic features
        stime = time.time()
        for udf in fnames:
            if udf in args.dynamic_features:
                self.count_features(udf)
        etime = time.time()
        sys.stderr.write("count_feature cost {} s...\n".format(etime-stime))

        # build feature2idx dict
        stime = time.time()
        for f in fnames:
            self.build_feature2idx(f)
        etime = time.time()
        sys.stderr.write("build_feature2idx cost {} s...\n".format(etime-stime))

        # save infos
        if self.is_train:
            self.save_max_idxs(fnames)

    
    # for batch load

    def build_nn_data(self, fnames, to_bin=False):
        stime = time.time()

        for key in fnames:
            if key == 'label':
                continue
            sys.stderr.write("building idx of {} for nn...\n".format(key))
            if key not in self.ids:
                self.ids[key] = []
                self.id_lens[key] = []
            self.max_lens[key] = 1
            for data in self.id_data[key]:
                tmp_ids = []
                for d in data.split(' '):
                    if d in self.feature2idx[key]:
                        tmp_ids.append(self.feature2idx[key][d])
                    else:
                        tmp_ids.append(1)
                dyl = len(tmp_ids)
                self.id_lens[key].append(dyl)
                self.ids[key].append(tmp_ids)
                if dyl > self.max_lens[key]:
                    self.max_lens[key] = dyl
            # padding
            for i in range(len(self.ids[key])):
                self.ids[key][i] = self.ids[key][i] + ([0]*(self.max_lens[key]-self.id_lens[key][i]))
            self.ids[key] = np.asarray(self.ids[key])
            self.id_lens[key] = np.asarray(self.id_lens[key])

            if to_bin:
                self.to_bin(key, self.n_parts)
            if to_bin:
                self.save_max_lens(key)

        if self.is_train:
            sys.stderr.write("building label for nn...\n")
            self.labels = [int(x) for x in self.id_data["label"]]
            self.labels = np.asarray(self.labels)
            if to_bin:
                self.to_bin("label", self.n_parts)

        etime = time.time()
        sys.stderr.write("build_nn_feature cost {} s...\n".format(etime-stime))


    def to_bin(self, fname, n_parts):
        for p in range(n_parts):
            sp = p * self.parts
            ep = (p+1) * self.parts
            if fname != "label":
                bin_file_path = os.path.join(args.root_data_path, 'bin_files/{}/{}_{}.bin'.format(self.type_name, fname, p))
                self.ids[fname][sp:ep].tofile(bin_file_path, format="%d")
                if fname in args.dynamic_features:
                    bin_file_path = os.path.join(args.root_data_path, 'bin_files/{}/{}_feature_cnt_{}.bin'.format(self.type_name, fname, p))
                    self.len_data[fname][sp:ep].tofile(bin_file_path, format="%d")
                    bin_file_path = os.path.join(args.root_data_path, 'bin_files/{}/{}_len_{}.bin'.format(self.type_name, fname, p))
                    self.id_lens[fname][sp:ep].tofile(bin_file_path, format="%d")
            else:
                bin_file_path = os.path.join(args.root_data_path, 'bin_files/{}/{}_{}.bin'.format(self.type_name, fname, p))
                self.labels[sp:ep].tofile(bin_file_path, format="%d")

    def load_bin(self, fnames, part):
        stime = time.time()
        self.load_max_lens(fnames, part)
        for key in fnames:
            self.from_bin(key, part)
        if self.is_train:
            self.from_bin("label", part)
        self.n_all_data = self.ids[fnames[0]].shape[0]
        etime = time.time()
        sys.stderr.write("load {} bin cost {} s\n".format(part, etime-stime))

    def from_bin(self, fname, part):
        if fname != "label":
            bin_file_path = os.path.join(args.root_data_path, 'bin_files/{}/{}_{}.bin'.format(self.type_name, fname, part))
            self.ids[fname] = np.fromfile(bin_file_path, dtype=int)
            if fname in args.dynamic_features:
                bin_file_path = os.path.join(args.root_data_path, 'bin_files/{}/{}_len_{}.bin'.format(self.type_name, fname, part))
                self.id_lens[fname] = np.fromfile(bin_file_path, dtype=int)
                self.ids[fname] = self.ids[fname].reshape(-1, self.max_lens[fname])
            else:
                self.ids[fname] = self.ids[fname].reshape(-1, 1)
            '''
            if fname in args.dynamic_features:
               bin_file_path = os.path.join(args.root_data_path, 'bin_files/{}/{}_feature_cnt_{}.bin'.format(self.type_name, fname, part))
               self.len_data[fname] = np.fromfile(bin_file_path, dtype=int)
            '''
        else:
            bin_file_path = os.path.join(args.root_data_path, 'bin_files/{}/{}_{}.bin'.format(self.type_name, fname, part))
            self.labels = np.fromfile(bin_file_path, dtype=int)


    def build_conversion_rate(self, fname):
        label_counts_file_path = os.path.join(args.root_data_path, 'infos/conversion_infos/label_counts.pkl')
        fname_counts_file_path = os.path.join(args.root_data_path, 'infos/conversion_infos/{}_counts.pkl'.format(fname))
        if os.path.exists(label_counts_file_path) and os.path.exists(fname_counts_file_path):
            self.label_counts = pickle.load(open(label_counts_file_path, 'r'))
            self.fname_counts = pickle.load(open(fname_counts_file_path, 'r'))
        else:
            self.label_counts = [0, 0]
            self.fname_counts = [{}, {}]
            for p in range(args.n_train_parts):
                self.load_bin([fname], p)
                data = self.ids[fname]
                label = self.labels
                # count fname in label==1 and label==0
                # count labels: label_counts[0] for label==0 and label_counts[1] for label==1
                for ds, l in zip(data,label):
                    idx = int(l)
                    self.label_counts[idx] += 1.0
                    for d in ds:
                        if d not in self.fname_counts[idx]:
                            self.fname_counts[idx][d] = 0
                        self.fname_counts[idx][d] += 1.0
            # save conversion_rate dict
            if self.is_train:
                pickle.dump(self.label_counts, open(label_counts_file_path, 'w'))
                pickle.dump(self.fname_counts, open(fname_counts_file_path, 'w'))

    def save_conversion_rate(self, fname, part):
        sys.stderr.write("saving {} conversion_rates part {}\n".format(fname, part))
        self.load_bin([fname], part)
        data = self.ids[fname]
        self.conversion_rates[fname] = []
        for ds in data:
            cr = [0.0, 0.0]
            cn = [0.00001, 0.00001]
            for d in ds:
                # sum(tf/tl * tf) / sum(tf)
                if d not in self.fname_counts[0]:
                    cr[0] += 0
                    cn[0] += 0
                else:
                    cr[0] += self.fname_counts[0][d] / self.label_counts[0] * self.fname_counts[0][d]
                    cn[0] += self.fname_counts[0][d]
                if d not in self.fname_counts[1]:
                    cr[1] += 0
                    cn[1] += 0
                else:
                    cr[1] += self.fname_counts[1][d] / self.label_counts[1] * self.fname_counts[1][d]
                    cn[1] += self.fname_counts[1][d]
            self.conversion_rates[fname].append([cr[0]/cn[0], cr[1]/cn[1]])

        # save to bin
        self.conversion_rates[fname] = np.asarray(self.conversion_rates[fname])

        bin_file_path = os.path.join(args.root_data_path, 'bin_files/{}/{}_conversion_rate_{}.bin'.format(self.type_name, fname, part))
        self.conversion_rates[fname].tofile(bin_file_path, format='%f')

    def load_counts(self, fnames):
        for fname in fnames:
            if fname in args.combine_features + args.len_static_features:
                continue
            fname_counts_file_path = os.path.join(args.root_data_path, 'infos/conversion_infos/{}_counts.pkl'.format(fname))
            self.feature_counts[fname] = pickle.load(open(fname_counts_file_path, 'r'))
            

    def load_conversion_rate_from_bin(self, fnames, part):
        for fname in fnames:
            bin_file_path = os.path.join(args.root_data_path, 'bin_files/{}/{}_conversion_rate_{}.bin'.format(self.type_name, fname, part))
            self.conversion_rates[fname] = np.fromfile(bin_file_path, dtype=float)
            self.conversion_rates[fname] = self.conversion_rates[fname].reshape(-1, 2)

    def random_shuffle(self, fnames):
        stime = time.time()
        rng_state = np.random.get_state()
        for key in fnames:
            np.random.set_state(rng_state)
            np.random.shuffle(self.ids[key])
            if key in args.dynamic_features:
                np.random.set_state(rng_state)
                np.random.shuffle(self.id_lens[key])
            if self.has_cr:
                np.random.set_state(rng_state)
                np.random.shuffle(self.conversion_rates[key])
        np.random.set_state(rng_state)
        np.random.shuffle(self.labels)
        etime = time.time()
        sys.stderr.write("random_shuffle cost {} s...\n".format(etime-stime))

    def reset(self):
        self.next_idx = 0

    def cut_threshold(self, fnames, ids, id_lens, threshold=500):
        for key in fnames:
            if key in args.combine_features+ args.len_static_features:
                continue
            new_ids = []
            for xs, ls in zip(ids[key], id_lens[key]):
                for i in range(ls):
                    tmp_cnt = 0
                    if xs[i] in self.feature_counts[key][0]:
                        tmp_cnt += self.feature_counts[key][0][xs[i]]
                    if xs[i] in self.feature_counts[key][1]:
                        tmp_cnt += self.feature_counts[key][1][xs[i]]
                    if tmp_cnt < threshold:
                        xs[i] = 1
                new_ids.append(xs)
            ids[key] = np.asarray(new_ids)
        return ids

    def fix_0to1(self, fnames, ids, id_lens):
        for key in fnames:
            new_ids = []
            for xs, ls in zip(ids[key], id_lens[key]):
                for i in range(ls):
                    #  if xs[i] == 0 or xs[i] >= self.max_idxs[key]:
                    if xs[i] == 0:
                        xs[i] = 1
                    elif xs[i] >= self.max_idxs[key]:
                        if 'len' in key:
                            xs[i] = self.max_idxs[key] - 1
                        else:
                            xs[i] = 1
                new_ids.append(xs)
            ids[key] = np.asarray(new_ids)
        return ids


    def next_batch(self, fnames_arr, batch_size):
        if self.next_idx >= self.n_all_data:
            return None
        end_idx = self.next_idx + batch_size
        st_ids = {}
        st_lens = {}
        dy_ids = {}
        dy_lens = {}
        conversion_rates = {}
        labels = self.labels[self.next_idx:end_idx]
        for key in fnames_arr[0]:
            st_ids[key] = self.ids[key][self.next_idx:end_idx]
            st_lens[key] = np.asarray([1]*len(st_ids[key]))
        for key in fnames_arr[1]:
            dy_ids[key] = self.ids[key][self.next_idx:end_idx]
            dy_lens[key] = self.id_lens[key][self.next_idx:end_idx]
        if self.has_cr:
            for key in fnames_arr[0]+fnames_arr[1]:
                conversion_rates[key] = self.conversion_rates[key][self.next_idx:end_idx]
        self.next_idx = end_idx
        # cut
        threshold=args.cut_threshold
        st_ids = self.cut_threshold(fnames_arr[0], st_ids, st_lens, threshold)
        dy_ids = self.cut_threshold(fnames_arr[1], dy_ids, dy_lens, threshold)
        # fix 0 to 1
        st_ids = self.fix_0to1(fnames_arr[0], st_ids, st_lens)
        dy_ids = self.fix_0to1(fnames_arr[1], dy_ids, dy_lens)
        return [st_ids, dy_ids, dy_lens, labels, conversion_rates]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--type_name", type=str, default="train_shuf", help="train_set name")
    parser_args = parser.parse_args()
    assert parser_args.type_name in ["train_shuf", "test1", "test2", 'chusai_train_shuf', 'valid', 'train_all']

    is_train = 'train' in parser_args.type_name or 'valid' in parser_args.type_name
    if is_train:
        args.root_data_path = args.root_train_data_path
    dataLoader = DataLoader(type_name=parser_args.type_name, is_train=is_train)

    #  now_features = args.user_static_features
    #  now_features = args.ad_static_features
    #  now_features = args.user_dynamic_features 
    now_features = args.ad_static_features + args.user_static_features + args.user_dynamic_features + ['uid']
    #  now_features = args.combine_features
    #  now_features = ['uid']
    #  now_features = ['label']

    # build all data from the beginning
    dataLoader.prepare_for_final_data(now_features)
    dataLoader.build_nn_data(now_features, to_bin=True)

