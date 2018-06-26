#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 ouwj <ouwenjie>
#
# Distributed under terms of the MIT license.

"""
build the embedding of dynamic matrix [Batch*Field_size*Dynamic_Feature_Size]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from args import *


class DyEmb(nn.Module):
    def __init__(self, fnames, max_idxs, embedding_size=4, dropout_rate=None, method='avg', use_cuda=True):
        """
        fnames: feature names
        max_idxs: array of max_idx of each feature
        embedding_size: size of embedding
        dropout: prob for dropout, set None if no dropout
        method: 'avg' or 'sum'
        use_cuda: bool, True for gpu or False for cpu
        """
        super(DyEmb, self).__init__()

        assert method in ['avg', 'sum']

        self.fnames = fnames
        self.max_idxs = max_idxs
        self.embedding_size = embedding_size
        self.dropout_rate = dropout_rate
        self.method = method
        self.use_cuda = use_cuda

        # initial layer
        self.embeddings = nn.ModuleList([nn.Embedding(max_idx, self.embedding_size, padding_idx=0) for max_idx in self.max_idxs])

        self.is_dropout = False
        if self.dropout_rate is not None:
            self.is_dropout = True
            self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, dynamic_ids, dynamic_lengths):
        """
        input: relative id 
        dynamic_ids: Batch_size * Field_size * Max_feature_size
        dynamic_lengths: Batch_size * Field_size 
        return: Batch_size * Field_size * Embedding_size
        """

        concat_embeddings = []
        for i, key in enumerate(self.fnames):
            # B*M
            dynamic_ids_tensor = torch.autograd.Variable(torch.LongTensor(dynamic_ids[key]))
            dynamic_lengths_tensor = torch.autograd.Variable(torch.FloatTensor(dynamic_lengths[key]))
            if self.use_cuda:
                dynamic_ids_tensor = dynamic_ids_tensor.cuda()
                dynamic_lengths_tensor = dynamic_lengths_tensor.cuda()

            batch_size = dynamic_ids_tensor.size()[0]
            max_feature_size = dynamic_ids_tensor.size()[-1]

            # embedding layer B*M*E
            dynamic_embeddings_tensor = self.embeddings[i](dynamic_ids_tensor)

            # dropout
            if self.is_dropout:
                dynamic_embeddings_tensor = self.dropout(dynamic_embeddings_tensor)

            # average B*M*E --AVG--> B*E
            dynamic_embedding = torch.sum(dynamic_embeddings_tensor, 1)

            if self.method == 'avg':
                # B*E -> B*1*E
                dynamic_lengths_tensor = dynamic_lengths_tensor.view(-1, 1).expand_as(dynamic_embedding)
                dynamic_embedding = dynamic_embedding / dynamic_lengths_tensor
            concat_embeddings.append(dynamic_embedding.view(batch_size, 1, self.embedding_size))
        # B*F*E
        concat_embeddings = torch.cat(concat_embeddings, 1)
        return concat_embeddings


if __name__ == '__main__':
    # test
    max_idxs = [4, 6]
    fnames = ["1", "2"]
    ids = {"1":[[2,1,3,0,0], [2,2,0,0,0]], "2":[[5,0,0,0,0],[5,5,5,5,5]]}
    lengths = {"1":[3,2], "2":[1,5]}


    dyEmb = DyEmb(fnames, max_idxs, use_cuda=False)

    avg_embeddings = dyEmb(ids, lengths)

    print avg_embeddings



