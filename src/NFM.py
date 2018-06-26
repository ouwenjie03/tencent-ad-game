#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 ouwj <ouwenjie>
#
# Distributed under terms of the MIT license.

"""
model:
    input: [single feature id] + [multi feature ids]
    embedding_layer: single : embedding    multi: avg_embedding
    lr_layer: one-hot lr + embeddings_concat lr
    activate_layer: sigmoid
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import time, sys

from staticEmbedding import *
from dynamicEmbedding import *
from support_model import *


class DyNFM(nn.Module):
    def __init__(self, batch_size, field_sizes, total_feature_sizes, embedding_size=4,
                 dropout_rate=None, batch_norm=True, use_cuda=True):
        """
        batch_size: batch_size
        field_sizes: length of feature_sizes, [n_single_field, n_multi_field]
        total_feature_sizes: total feature size, [n_single_feature, n_multi_feature]
        embedding_sizes: size of embedding, [n_single_embedding, n_multi_embedding]
        dropout_rate: prob for dropout, set None if no dropout,
        use_cuda: bool, True for gpu or False for cpu
        """
        if batch_norm:
            dropout_rate = None

        super(DyNFM, self).__init__()

        self.batch_size = batch_size
        self.field_sizes = field_sizes
        self.total_feature_sizes = total_feature_sizes
        self.embedding_size = embedding_size
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.use_cuda = use_cuda

        self.stEmb = StEmb(self.batch_size, self.field_sizes[0], self.total_feature_sizes[0], embedding_size=self.embedding_size, dropout_rate=self.dropout_rate, use_cuda=self.use_cuda)
        self.dyEmb = DyEmb(self.batch_size, self.field_sizes[1], self.total_feature_sizes[1], embedding_size=self.embedding_size, dropout_rate=self.dropout_rate, method='avg', use_cuda=self.use_cuda)

        self.stLr = StEmb(self.batch_size, self.field_sizes[0], self.total_feature_sizes[0], embedding_size=1, dropout_rate=self.dropout_rate, use_cuda=self.use_cuda)
        self.dyLr = DyEmb(self.batch_size, self.field_sizes[1], self.total_feature_sizes[1], embedding_size=1, dropout_rate=self.dropout_rate, method='sum', use_cuda=self.use_cuda)

        self.bias = torch.nn.Parameter(torch.randn(1))

        '''
        if self.use_cuda:
            self.stEmb = self.stEmb.cuda()
            self.dyEmb = self.dyEmb.cuda()
            self.stLr = self.stLr.cuda()
            self.dyLr = self.dyLr.cuda()
        '''

        # embedding lr layer, 3 layers
        self.all_field_size = self.field_sizes[0] + self.field_sizes[1]
        self.embLr_input_dim = self.all_field_size * (self.all_field_size - 1) // 2 * self.embedding_size
        self.hidden_size_1 = 512
        self.batch_norm_1 =  nn.BatchNorm1d(self.embLr_input_dim)
        self.embLr1 = nn.Linear(self.embLr_input_dim, self.hidden_size_1)
        self.batch_norm_2 = nn.BatchNorm1d(self.hidden_size_1)
        self.hidden_size_2 = 512
        self.embLr2 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.batch_norm_3 = nn.BatchNorm1d(self.hidden_size_2)
        self.hidden_size_3 = 512
        self.embLr3 = nn.Linear(self.hidden_size_2, self.hidden_size_3)

        self.batch_norm_out = nn.BatchNorm1d(self.hidden_size_3)
        self.embLr_out = nn.Linear(self.hidden_size_3, 1)

        self.embLr_is_dropout = False
        if self.dropout_rate is not None:
            self.embLr_is_dropout = True
            self.emb_dropout = nn.Dropout(self.dropout_rate)

        # mask for combination
        self.mask_len = [x + 1 for x in range(self.all_field_size)]
        self.mask = make_mask(self.mask_len, self.all_field_size, fill_val=False)
        if self.use_cuda:
            self.mask = self.mask.cuda()

    def forward(self, static_ids, dynamic_ids, dynamic_lengths):
        """
        input: relative id
        static_ids: Batch_size * Field_size
        dynamic_ids: Batch_size * Field_size * Max_feature_size
        dynamic_lengths: Batch_size * Field_size
        return: Batch_size * 1,  probs
        """

        # embedding layers
        dynamic_embeddings = self.dyEmb(dynamic_ids, dynamic_lengths)

        static_embeddings = self.stEmb(static_ids)

        batch_size = static_embeddings.size()[0]

        # B*F*E
        all_embeddings = torch.cat([static_embeddings, dynamic_embeddings], 1)
        #  field_size = all_embeddings.size()[1]
        field_size = self.field_sizes[0] + self.field_sizes[1]

        # combine feature by multi
        '''
        self.combine_embeddings = []
        for i in range(self.all_embeddings.size()[1]):
            for j in range(i+1, self.all_embeddings.size()[1]):
                new_emb = self.all_embeddings[:,i,:] * self.all_embeddings[:,j,:]
                self.combine_embeddings.append(new_emb)

        self.combine_embeddings = torch.cat(self.combine_embeddings, 1)
        '''
        all_embeddings_row = all_embeddings.view(batch_size, 1, field_size, self.embedding_size).expand(batch_size, field_size, field_size, self.embedding_size)
        all_embeddings_col = all_embeddings.view(batch_size, field_size, 1, self.embedding_size).expand(batch_size, field_size, field_size, self.embedding_size)
        all_embeddings_combine = all_embeddings_col * all_embeddings_row
        all_mask = self.mask.view(1, field_size, field_size, 1).expand(batch_size, field_size, field_size, self.embedding_size)
        combine_embeddings = torch.masked_select(all_embeddings_combine, all_mask)

        # lr layer
        static_lr_out = self.stLr(static_ids).view(batch_size, -1)

        dynamic_lr_out = self.dyLr(dynamic_ids, dynamic_lengths).view(batch_size, -1)

        # embedding lr layer
        # B*F1*E + B*F2*E -> B*[F1+F2]*E -> B*[F*E]
        embedding_lr_in_1 = combine_embeddings.view(batch_size, -1)
        if self.embLr_is_dropout:
            embedding_lr_in_1 = self.emb_dropout(embedding_lr_in_1)
        if self.batch_norm:
            embedding_lr_in_1 = self.batch_norm_1(embedding_lr_in_1)
        embedding_lr_out_1 = self.embLr1(embedding_lr_in_1)

        embedding_lr_in_2 = F.relu(embedding_lr_out_1)
        if self.embLr_is_dropout:
            embedding_lr_in_2 = self.emb_dropout(embedding_lr_in_2)
        if self.batch_norm:
            embedding_lr_in_2 = self.batch_norm_2(embedding_lr_in_2)
        embedding_lr_out_2 = self.embLr2(embedding_lr_in_2)

        embedding_lr_in_3 = F.relu(embedding_lr_out_2)
        if self.embLr_is_dropout:
            embedding_lr_in_3 = self.emb_dropout(embedding_lr_in_3)
        if self.batch_norm:
            embedding_lr_in_3 = self.batch_norm_3(embedding_lr_in_3)
        embedding_lr_out_3 = self.embLr3(embedding_lr_in_3)

        embedding_lr_in = F.relu(embedding_lr_out_3)
        if self.embLr_is_dropout:
            embedding_lr_in = self.emb_dropout(embedding_lr_in)
        if self.batch_norm:
            embedding_lr_in = self.batch_norm_out(embedding_lr_in)
        embedding_lr_out = self.embLr_out(embedding_lr_in)

        # output
        #  print self.static_lr_out
        #  print self.dynamic_lr_out
        #  print self.embedding_lr_out
        scores = self.bias + torch.sum(static_lr_out, -1) + torch.sum(dynamic_lr_out, -1) + torch.sum(embedding_lr_out_3, -1)

        # activate layer
        # self.probs = F.sigmoid(self.scores)

        return scores

    def get_loss(self, scores, labels):
        """
        binary cross entropy loss
        """
        labels = torch.autograd.Variable(labels, requires_grad=False)
        loss = F.binary_cross_entropy_with_logits(scores, labels)
        return loss


if __name__ == '__main__':
    batch_size = 2

    dy_field_size = 2
    dy_total_feature_size = 8
    dy_ids = torch.LongTensor([[[2, 1, 3, 0, 0], [5, 0, 0, 0, 0]], [[2, 2, 0, 0, 0], [5, 5, 5, 5, 5]]])
    dy_lengths = torch.LongTensor([[3, 1], [2, 5]])

    st_field_size = 2
    st_total_feature_size = 6
    st_ids = torch.LongTensor([[1, 5], [2, 5]])

    reals = torch.FloatTensor([1, 0])

    dyNfm = DyNFM(batch_size, [st_field_size, dy_field_size], [st_total_feature_size, dy_total_feature_size], use_cuda=False)
    #  dyNfm = dyNfm.cuda()

    probs = dyNfm(st_ids, dy_ids, dy_lengths)
    print(probs)

    loss = dyNfm.get_loss(probs, reals)
    print(loss)

