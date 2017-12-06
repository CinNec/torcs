# MIT License
#
# Copyright (c) 2017 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Tom Runia
# Date Created: 2017-10-19

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

class LSTM(object):

    def __init__(self, input_length, input_dim, num_hidden, num_classes, batch_size):

        self._input_length = input_length
        self._input_dim    = input_dim
        self._num_hidden   = num_hidden
        self._num_classes  = num_classes
        self._batch_size   = batch_size

        self.initializer_weights = tf.variance_scaling_initializer()
        self.initializer_biases  = tf.constant_initializer(0.0)

        # Initialize the stuff you need
        # ...
        self.x = tf.placeholder(tf.uint8, shape=[self._batch_size, self._input_length - 1], name='inputs')
        self.y = tf.placeholder(tf.uint8, shape=[self._batch_size], name='outputs')

        self.x_oh = tf.one_hot(self.x, self._num_classes)
        self.y_oh = tf.one_hot(self.y, self._num_classes)

        self.W_gx, self.W_gh, self.b_g = self.get_gate('g')
        self.W_ix, self.W_ih, self.b_i = self.get_gate('i')
        self.W_fx, self.W_fh, self.b_f = self.get_gate('f')
        self.W_ox, self.W_oh, self.b_o = self.get_gate('o')

        self.W_out = tf.get_variable(name='W_out',
                                    shape=[num_hidden, num_classes],
                                    initializer = self.initializer_weights,
                                    regularizer=None)
        self.b_out = tf.get_variable(name='b_out',
                                    shape=[num_classes],
                                    initializer = self.initializer_biases)


    def get_gate(self, name):
        W_x = tf.get_variable(name='W_{}x'.format(name),
                                    shape=[self._num_classes, self._num_hidden],
                                    initializer=self.initializer_weights,
                                    regularizer=None)
        W_h = tf.get_variable(name='W_{}h'.format(name),
                                    shape=[self._num_hidden, self._num_hidden],
                                    initializer=self.initializer_weights,
                                    regularizer=None)
        b =  tf.get_variable(name='b_{}'.format(name),
                                    shape=[self._num_hidden],
                                    initializer = self.initializer_biases)

        return W_x, W_h, b

    def _lstm_step(self, lstm_state_tuple, x):
        # Single step through LSTM cell ...
        c, h = tf.unstack(lstm_state_tuple)

        print('-----')
        print('Shapes')
        print('Wgx: {}, x: {}, W_gh: {}, h: {}, b_g: {}\n'.format(self.W_gx.get_shape(), np.shape(x), self.W_gh.get_shape(), np.shape(h),self.b_g.get_shape()))

        print('First wgx matmul x : {}, wgh matmul h : {}\n ----------------'.format(tf.matmul(x, self.W_gx), tf.matmul(self.W_gh, h)))


        g = tf.nn.tanh(tf.matmul(x, self.W_gx) + tf.matmul(h, self.W_gh) + self.b_g)

        i = tf.nn.sigmoid(tf.matmul(x, self.W_ix) + tf.matmul(h, self.W_ih) + self.b_i)
        f = tf.nn.sigmoid(tf.matmul(x, self.W_fx) + tf.matmul(h, self.W_fh) + self.b_f)
        o = tf.nn.sigmoid(tf.matmul(x, self.W_ox) + tf.matmul(h, self.W_oh) + self.b_o)

        c = (g * i) + (c * f)

        h = tf.nn.tanh(c) * o

        return tf.stack([c,h])

    def compute_logits(self):
        # Implement the logits for predicting the last digit in the palindrome
        inputs = tf.unstack(self.x_oh, axis=0)

        h_T = tf.scan(fn=self._lstm_step,
                      elems=inputs,
                      initializer=tf.zeros([2, self._batch_size, self._num_hidden]))

        logits = tf.matmul(h_T[-1][1], self.W_out) + self.b_out

        self._logits = logits


        return logits

    def compute_loss(self):
        # Implement the cross-entropy loss for classification of the last digit
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_oh, logits=self._logits))

        tf.summary.scalar('cross_entropy', loss)
        return loss

    def accuracy(self):
        # Implement the accuracy of predicting the
        # last digit over the current batch ...
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.y_oh, 1), tf.argmax(self._logits, 1)), tf.float32))

        tf.summary.scalar('accuracy', accuracy)
        return accuracy

    @property
    def inputs(self):
        return self.x

    @property
    def outputs(self):
        return self.y
