# Adapted from CS-DRO (KDD 2026), https://github.com/gyeomo/CS-DRO.
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# See CS_DRO_LICENSE.txt for the upstream MIT license.

import numpy as np
import torch
import torch.nn as nn


class LightEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size, num_hidden_layers=0) -> None:
        super(LightEncoder, self).__init__()
        self.dropout = nn.Dropout(0.25)
        layers = [
            nn.Linear(in_channels, hidden_size),
            # nn.GELU(),
            # nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        ]
        self.encoder = nn.Sequential(*layers)
        self._initialize_weights(self.encoder)

    def _initialize_weights(self, modules):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        feat = self.encoder(x)
        return feat


class NotearsClassifier(nn.Module):
    def __init__(self, dims, num_classes):
        super(NotearsClassifier, self).__init__()
        self.dims = dims
        self.num_classes = num_classes
        self.weight_pos = nn.Parameter(torch.zeros(dims + 1, dims + 1))
        self.weight_neg = nn.Parameter(torch.zeros(dims + 1, dims + 1))
        self.register_buffer("_I", torch.eye(dims + 1))
        self.register_buffer("_repeats", torch.ones(dims + 1).long())
        self._repeats[-1] *= num_classes

    def _adj(self):
        return self.weight_pos - self.weight_neg

    def _get_matrix(self):
        W = self._adj()
        W = torch.sqrt(W * W + 1e-10)[:self.dims, -1].detach()
        return W

    def _get_matrix_reverse(self):
        W = self._adj()#.T
        W = torch.sqrt(W * W + 1e-10)[-1,:self.dims].detach()
        return W

    def _adj_sub(self):
        W = self._adj()
        return torch.sqrt(W * W + 1e-10)

    def h_func(self):
        W = self._adj()
        E = torch.matrix_exp(W * W)
        h = torch.trace(E) - self.dims - 1
        return h

    def w_l1_reg(self):
        reg = torch.sum(torch.abs(self.weight_pos + self.weight_neg))
        return reg

    def forward(self, x, y=None):
        W = self._adj()
        W_sub = self._adj_sub()
        if y is not None:
            x_aug = torch.cat((x, y.unsqueeze(1)), dim=1)
            M = x_aug @ W
            masked_x = x * W_sub[:self.dims, -1].unsqueeze(0)
            # reconstruct variables, classification logits
            return M[:, :self.dims], masked_x
        else:
            masked_x = x * W[:self.dims, -1].unsqueeze(0).detach()
            return masked_x

    def mask_feature(self, x, flag):
        if flag == "class":
            W_sub = self._adj_sub()
        else:
            W_sub = self._adj_sub_reverse()
        mask = W_sub[:self.dims, -1].unsqueeze(0).detach()
        return x * mask

    @torch.no_grad()
    def projection(self):
        self.weight_pos.data.clamp_(0, None)
        self.weight_neg.data.clamp_(0, None)
        self.weight_pos.data.fill_diagonal_(0)
        self.weight_neg.data.fill_diagonal_(0)

    @torch.no_grad()
    def masked_ratio(self):
        W = self._adj()
        return torch.norm(W[:self.dims, -1], p=0)
