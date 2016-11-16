import os
import tensorflow as tf

from .layers import *
from .network import Network

class RNNCNN(Network):
    def __init__(self, sess,
                 data_format,
                 history_length,
                 num_steps,
                 num_layers,
                 observation_dims,
                 output_size,
                 trainable=True,
                 hidden_activation_fn=tf.nn.relu,
                 output_activation_fn=None,
                 weights_initializer=initializers.xavier_initializer(),
                 biases_initializer=tf.constant_initializer(0.1),
                 value_hidden_sizes=[512],
                 advantage_hidden_sizes=[512],
                 network_output_type='dueling',
                 network_header_type='nips',
                 name='CNN'):
        super(RNNCNN, self).__init__(sess, name)

        if data_format == 'NHWC':
            self.inputs = tf.placeholder('float32',
                    [None] + observation_dims + [history_length], name='inputs')
        elif data_format == 'NCHW':
            self.inputs = tf.placeholder('float32',
                    [None, history_length] + observation_dims, name='inputs')
        else:
            raise ValueError("unknown data_format : %s" % data_format)

        self.var = {}
        self.l0s = tf.div(self.inputs, 255.)
        self.l0s = tf.split(1, num_steps, self.l0s)

        layers = []
        with tf.variable_scope(name):
            for t,l0 in enumerate(self.l0s):
                # TODO: not sure why get_variable is not just reusing variables
                if t > 0: tf.get_variable_scope().reuse_variables()
                self.l1, self.var['l1_w'], self.var['l1_b'] = conv2d(l0,
                        16, [8, 8], [4, 4], weights_initializer, biases_initializer,
                        hidden_activation_fn, data_format, name='l1_conv')
                self.l2, self.var['l2_w'], self.var['l2_b'] = conv2d(self.l1,
                        32, [4, 4], [2, 2], weights_initializer, biases_initializer,
                        hidden_activation_fn, data_format, name='l2_conv')
                self.l3, self.var['l3_w'], self.var['l3_b'] = \
                        linear(self.l2, 256, weights_initializer, biases_initializer,
                        hidden_activation_fn, data_format, name='l3_conv')
                layers.append(self.l3)

        with tf.variable_scope(name):
            self.cell = tf.nn.rnn_cell.LSTMCell(256)
            self.cell = tf.nn.rnn_cell.MultiRNNCell([self.cell]*num_layers)
            outputs, state = tf.nn.dynamic_rnn(self.cell, tf.pack(layers), dtype=tf.float32,
                                               time_major=True)
            layer = outputs[-1]

            self.build_output_ops(layer, network_output_type,
                    value_hidden_sizes, advantage_hidden_sizes, output_size,
                    weights_initializer, biases_initializer, hidden_activation_fn,
                    output_activation_fn, trainable)
