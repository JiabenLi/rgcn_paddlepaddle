#!/usr/bin/env python
# coding: utf-8

import paddle.nn as nn
import paddle.nn.functional as F
from layers import *


class RelationalGraphConvModel(nn.Layer):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_bases,
        num_rel,
        num_layer,
        dropout,
        featureless=True,
        cuda=False,
    ):
        super(RelationalGraphConvModel, self).__init__()

        self.num_layer = num_layer
        self.dropout = dropout
        self.layers = nn.LayerList()
        self.relu = nn.ReLU()

        for i in range(self.num_layer):
            if i == 0:
                self.layers.append(
                    RelationalGraphConvLayer(
                        input_size,
                        hidden_size,
                        num_bases,
                        num_rel,
                        bias=False,
                        cuda=cuda,
                    )
                )
            else:
                if i == self.num_layer - 1:
                    self.layers.append(
                        RelationalGraphConvLayer(
                            hidden_size,
                            output_size,
                            num_bases,
                            num_rel,
                            bias=False,
                            cuda=cuda,
                        )
                    )
                else:
                    self.layers.append(
                        RelationalGraphConvLayer(
                            hidden_size,
                            hidden_size,
                            num_bases,
                            num_rel,
                            bias=False,
                            cuda=cuda,
                        )
                    )

    def forward(self, A, X):
        # x = X
        x = None  # featureless
        for i, layer in enumerate(self.layers):
            x = layer(A, x)
            if i != self.num_layer - 1:
                x = F.dropout(self.relu(x), self.dropout, training=self.training)
            else:
                x = F.dropout(x, self.dropout, training=self.training)
        return x
