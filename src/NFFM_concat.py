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
from focal_loss import *
from support_model import *


class DyNFFM_concat(nn.Module):
    def __init__(self, fnames, max_idxs, embedding_size=4, dropout_rate=None, batch_norm=True, use_cuda=True):
        """
        fnames: feature names: [static feature names, dynamic feature names]
        max_idxs: max_idxs: [static max_idxs, dynamic max_idxs]
        embedding_sizes: size of embedding, [n_single_embedding, n_multi_embedding]
        dropout_rate: prob for dropout, set None if no dropout,
        use_cuda: bool, True for gpu or False for cpu
        """

        super(DyNFFM_concat, self).__init__()
        self.fnames = fnames
        self.max_idxs = max_idxs
        self.n_fnames = len(fnames[0]) + len(fnames[1])
        self.embedding_size = embedding_size
        self.field_embedding_size = embedding_size * self.n_fnames
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.use_cuda = use_cuda

        self.stEmb = StEmb(
                self.fnames[0],
                self.max_idxs[0],
                embedding_size=self.field_embedding_size,
                dropout_rate=self.dropout_rate,
                use_cuda=self.use_cuda
                )
        self.dyEmb = DyEmb(
                self.fnames[1],
                self.max_idxs[1],
                embedding_size=self.field_embedding_size,
                dropout_rate=self.dropout_rate,
                method='avg',
                use_cuda=self.use_cuda
                )

        #  self.stLr = StEmb(
        #          self.fnames[0],
        #          self.max_idxs[0],
        #          embedding_size=1,
        #          dropout_rate=self.dropout_rate,
        #          use_cuda=self.use_cuda
        #          )

        #  self.dyLr = DyEmb(
        #          self.fnames[1],
        #          self.max_idxs[1],
        #          embedding_size=1,
        #          dropout_rate=self.dropout_rate,
        #          method='avg',
        #          use_cuda=self.use_cuda
        #          )

        self.bias = torch.nn.Parameter(torch.zeros(1))

        # focal loss layer
        self.fc_loss = FocalLoss(gamma=2)

        # mask for combination
        self.mask = []
        all_fnames = fnames[0] + fnames[1]
        self.n_bi = 0
        for i in range(self.n_fnames):
            tmp = []
            for j in range(self.n_fnames):
                if i >= j:
                #  if i >= j or filter_same_features(all_fnames[i], all_fnames[j]):
                    tmp.append(0)
                else:
                    tmp.append(1)
                    self.n_bi += 1
            self.mask.append(tmp)
        self.mask = torch.autograd.Variable(torch.ByteTensor(self.mask))

        # embedding lr layer
        # bi-interation + embedding concat
        self.embLr_input_dim = self.n_bi * self.embedding_size + self.n_fnames * self.n_fnames * self.embedding_size 
        self.hidden_sizes = [self.embLr_input_dim, 512, 256, 256, 128]
        self.n_linear_layers = len(self.hidden_sizes) - 1
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(self.hidden_sizes[i]) for i in range(self.n_linear_layers)])
        self.embLrs = nn.ModuleList([nn.Linear(self.hidden_sizes[i], self.hidden_sizes[i+1]) for i in range(self.n_linear_layers)])

        self.batch_norm_out = nn.BatchNorm1d(self.hidden_sizes[-1])
        self.embLr_out = nn.Linear(self.hidden_sizes[-1], 1)

        self.embLr_is_dropout = False
        if self.dropout_rate is not None:
            self.embLr_is_dropout = True
            self.emb_dropout = nn.Dropout(self.dropout_rate)


        #  self.mask_len = [x + 1 for x in range(self.n_fnames)]
        if self.use_cuda:
            self.mask = self.mask.cuda()
            self.fc_loss = self.fc_loss.cuda()
    
    def load(self, model_path):
        self.load_state_dict(torch.load(model_path))

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)


    def forward(self, static_ids, dynamic_ids, dynamic_lengths, conversion_rates):
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
        all_embeddings = all_embeddings.view(batch_size, self.n_fnames, self.n_fnames, self.embedding_size)

        # combine feature by multi
        all_mask = self.mask.view(1, self.n_fnames, self.n_fnames, 1).expand_as(all_embeddings)
        all_embeddings_ur = torch.masked_select(all_embeddings, all_mask)
        all_embeddings_1 = all_embeddings.transpose(2, 1)
        all_embeddings_ll = torch.masked_select(all_embeddings_1, all_mask)

        bi_embeddings = all_embeddings_ur * all_embeddings_ll

        # lr layer
        # static_lr_out = self.stLr(static_ids).view(batch_size, -1)
        # dynamic_lr_out = self.dyLr(dynamic_ids, dynamic_lengths).view(batch_size, -1)

        # embedding lr layer
        # B*F1*E + B*F2*E -> B*[F1+F2]*E -> B*[F*E]

        lr_out = torch.cat([bi_embeddings.view(batch_size, -1), all_embeddings.view(batch_size, -1)], -1)
        for i in range(self.n_linear_layers):
            lr_in = lr_out
            if self.batch_norm:
                lr_in = self.batch_norms[i](lr_in)
            if self.embLr_is_dropout:
                self.emb_dropout(lr_in)
            lr_out = self.embLrs[i](lr_in)
            lr_out = F.relu(lr_out)
        embedding_lr_in = lr_out

        if self.embLr_is_dropout:
            embedding_lr_in = self.emb_dropout(embedding_lr_in)
        if self.batch_norm:
            embedding_lr_in = self.batch_norm_out(embedding_lr_in)
        embedding_lr_out = self.embLr_out(embedding_lr_in)

        # output
        #  print self.static_lr_out
        #  print self.dynamic_lr_out
        #  print self.embedding_lr_out
        #  scores = self.bias + torch.sum(static_lr_out, -1) + torch.sum(dynamic_lr_out, -1) + torch.sum(embedding_lr_out, -1)
        scores = self.bias + torch.sum(embedding_lr_out, -1)

        # activate layer
        # self.probs = F.sigmoid(self.scores)

        return scores

    def get_loss(self, scores, labels):
        """
        binary cross entropy loss
        """
        labels = torch.autograd.Variable(torch.FloatTensor(labels), requires_grad=False)
        if self.use_cuda:
            labels = labels.cuda()

        #  BCE loss
        loss = F.binary_cross_entropy_with_logits(scores, labels)

        #  weighted BCE loss
        #  weights = labels * 10.0
        #  weights = weights.masked_fill_(labels.le(0.5), 1.0)
        #  loss = F.binary_cross_entropy_with_logits(scores, labels, weights)

        #  margin loss
        #  labels = labels.masked_fill_(labels.le(0.5), -1)
        #  loss = F.soft_margin_loss(scores, labels)

        #  focal loss
        #  scores = torch.sigmoid(scores).view(-1, 1)
        #  scores = torch.cat([1.0-scores, scores], -1)
        #  loss = self.fc_loss(scores, labels.long())

        return loss


if __name__ == '__main__':
    st_max_idxs = [4, 6]
    st_fnames = ["1", "2"]
    st_ids = {"1":[[2], [3]], "2":[[5],[1]]}

    dy_max_idxs = [4, 6]
    dy_fnames = ["1", "2"]
    dy_ids = {"1":[[2,1,3,0,0], [2,2,0,0,0]], "2":[[5,0,0,0,0],[5,5,5,5,5]]}
    dy_lengths = {"1":[3,1], "2":[2,5]}

    reals = [1, 0]

    dyNffm = DyNFFM_concat([st_fnames, dy_fnames], [st_max_idxs, dy_max_idxs], use_cuda=True)
    dyNffm.cuda()

    probs = dyNffm(st_ids, dy_ids, dy_lengths)
    print(probs)

    loss = dyNffm.get_loss(probs, reals)
    print(loss)

