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

import numpy as np
import tensorflow as tf

################################################################################

class VanillaRNN(object):

    def __init__(self, input_length, input_dim, num_hidden, num_classes, batch_size):

        self._input_length = input_length
        self._input_dim    = input_dim
        self._num_hidden   = num_hidden
        self._num_classes  = num_classes
        self._batch_size   = batch_size

        initializer_weights = tf.variance_scaling_initializer()
        initializer_biases  = tf.constant_initializer(0.0)

        # Initialize the stuff you need
        # ...
        self.x = tf.placeholder(tf.uint8, shape=[self._batch_size, self._input_length - 1], name='inputs')
        self.y = tf.placeholder(tf.uint8, shape=[self._batch_size], name='outputs')

        self.x_oh = tf.one_hot(self.x, self._num_classes)
        self.y_oh = tf.one_hot(self.y, self._num_classes)

        print('Input length: {}\nInput dim: {}\nNum hidden: {}\nNum classes: {}\nBatch size: {} \n'.format(input_length, input_dim, num_hidden, num_classes, batch_size))

        self.W_hx = tf.get_variable(name='W_hx',
                                    shape=[num_classes, num_hidden],
                                    initializer = initializer_weights,
                                    regularizer=None)

        self.W_hh = tf.get_variable(name='W_hh',
                                    shape=[num_hidden, num_hidden],
                                    initializer = initializer_weights,
                                    regularizer=None)

        self.b_h = tf.get_variable(name='b_h',
                                    shape=[num_hidden],
                                    initializer = initializer_biases)

        self.W_ho = tf.get_variable(name='W_ho',
                                    shape=[num_hidden, num_classes],
                                    initializer = initializer_weights,
                                    regularizer=None)

        self.b_o = tf.get_variable(name='b_o',
                                    shape=[num_classes],
                                    initializer = initializer_biases)

    def _rnn_step(self, h_prev, x):
        # Single step through Vanilla RNN cell ...
        print('Shape W_hx: {}, shape x: {}'.format(np.shape(self.W_hx), np.shape(x)))
        print('First wgx matmul x : {}, wgh matmul h : {}\n ----------------'.format(tf.matmul(self.W_hx, x), tf.matmul(self.W_hh, h_prev)))

        return tf.nn.tanh(tf.matmul(x, self.W_hx) + tf.matmul(h_prev, self.W_hh) + self.b_h)

    def compute_logits(self):
        # Implement the logits for predicting the last digit in the palindrome
        inputs = tf.unstack(self.x_oh, axis=0)

        h_T = tf.scan(fn=self._rnn_step,
                      elems=inputs,
                      initializer=tf.zeros(shape=[self._batch_size, self._num_hidden]))

        logits = tf.matmul(h_T[-1], self.W_ho) + self.b_o

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
