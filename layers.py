#!/usr/bin/env python
# coding: utf-8

import paddle
import paddle.nn as nn
import numpy as np
import paddlenlp

class RelationalGraphConvLayer(nn.Layer):
    def __init__(
        self, input_size, output_size, num_bases, num_rel, bias=False, cuda=False
    ):
        super(RelationalGraphConvLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_bases = num_bases
        self.num_rel = num_rel
        self.cuda = cuda

        # R-GCN weights
        if num_bases > 0:
            self.w_bases = paddle.static.create_parameter(
                shape=[self.num_bases, self.input_size, self.output_size],
                dtype='float32',
                default_initializer=paddle.nn.initializer.XavierUniform()
            )
            self.w_rel = paddle.static.create_parameter(
                shape=[self.num_rel, self.num_bases],
                dtype='float32',
                default_initializer=paddle.nn.initializer.XavierUniform()
            )
        else:
            self.w = paddle.static.create_parameter(
                shape=[self.num_rel, self.input_size, self.output_size],
                dtype='float32',
                default_initializer=paddle.nn.initializer.XavierUniform()
            )
        # R-GCN bias
        if bias:
            self.bias = paddle.static.create_parameter(
                shape=[self.output_size],
                dtype='float32',
                default_initializer=paddle.nn.initializer.XavierUniform()
            )
        else:
            self.bias = None

    def forward(self, A, X):
        self.w = (
            paddlenlp.ops.einsum("rb, bio -> rio", (self.w_rel, self.w_bases))
            if self.num_bases > 0
            else self.w
        )  # shape(r, input_size, output_size)
        weights = self.w.clone()
        # Each relations * Weight
        out = paddle.zeros(shape=[8285, self.output_size], dtype='float32')
        for i in range(self.num_rel):
            if X is not None:
                tmp = paddle.matmul(paddle.to_tensor(A[i].toarray(), dtype='float32'), X)
                out += paddle.matmul(tmp, weights[i])  # shape(#node, output_size)
            else:
                out += paddle.matmul(paddle.to_tensor(A[i].toarray(), dtype='float32'), weights[i])  # shape(#node, output_size)

        if self.bias is not None:
            out += self.bias.unsqueeze(0)
        return out