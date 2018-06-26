#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 ouwj <ouwenjie>
#
# Distributed under terms of the MIT license.

"""
build the embedding of static matrix [Batch*Field_size]
"""

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

class StEmb(nn.Module):
    def __init__(self, fnames, max_idxs, embedding_size=4, dropout_rate=None, use_cuda=True):
        """
        fnames: feature names
        max_idxs: array of max_idx of each feature
        embedding_size: size of embedding
        dropout: prob for dropout, set None if no dropout
        use_cuda: bool, True for gpu or False for cpu
        """
        super(StEmb, self).__init__()
        self.fnames = fnames
        self.max_idxs = max_idxs
        self.embedding_size = embedding_size
        self.dropout_rate = dropout_rate
        self.use_cuda = use_cuda

        # initial layer
        self.embeddings = nn.ModuleList([nn.Embedding(max_idx, self.embedding_size, padding_idx=0) for max_idx in self.max_idxs])

        self.is_dropout = False
        if self.dropout_rate is not None:
            self.is_dropout = True
            self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, static_ids):
        """
        input: relative id 
        static_ids: Batch_size * Field_size
        return: Batch_size * Field_size * Embedding_size
        """

        concat_embeddings = []
        #  sys.stderr.write('{}\n'.format([static_ids[k].shape for k in self.fnames]))
        for i, key in enumerate(self.fnames):
            # B*1
            static_ids_tensor = torch.autograd.Variable(torch.LongTensor(static_ids[key]))
            if self.use_cuda:
                static_ids_tensor = static_ids_tensor.cuda()

            # embedding layer B*1*E
            static_embeddings_tensor = self.embeddings[i](static_ids_tensor)

            # dropout
            if self.is_dropout:
                static_embeddings_tensor = self.dropout(static_embeddings_tensor)
            
            concat_embeddings.append(static_embeddings_tensor)
        # B*F*E
        concat_embeddings = torch.cat(concat_embeddings, 1)

        return concat_embeddings


if __name__ == '__main__':
    # test
    max_idxs = [4, 6]
    fnames = ["1", "2"]
    ids = {"1":[[2], [3]], "2":[[5],[1]]}

    stEmb = StEmb(fnames, max_idxs, use_cuda=False)

    st_embeddings = stEmb(ids)

    print st_embeddings



