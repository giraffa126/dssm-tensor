#!/usr/bin/env python
# coding: utf8
import sys
import os
import tensorflow as tf
import numpy as np

def cosine_similarity(x1, x2, eps=1e-12):
    w1 = tf.sqrt(tf.reduce_sum(x1 ** 2, axis=1))
    w2 = tf.sqrt(tf.reduce_sum(x2 ** 2, axis=1))
    w12 = tf.reduce_sum(x1 * x2, axis=1)
    return (w12 / (w1 * w2 + eps)) * 5

def rank_loss(left, right):
    pred_diff = left - right
    loss = tf.log1p(tf.exp(pred_diff)) - pred_diff
    return tf.reduce_mean(loss)


class DSSMNet(object):
    """ sim net
    """
    def __init__(self, vocab_size=None, embedding_size=128, hidden_size=256, eps=0.2):
        self.eps = eps
        # inputs
        self.query_in = tf.placeholder(tf.int32, [None, None], name="query")
        self.pos_in = tf.placeholder(tf.int32, [None, None], name="pos")
        self.neg_in = tf.placeholder(tf.int32, [None, None], name="neg")
        # use predict query-query similarity
        self.fake_in = tf.placeholder(tf.int32, [None, None], name="fake")

        with tf.variable_scope("embedding", reuse=tf.AUTO_REUSE):
            self.embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -0.1, 0.1), 
                    trainable=True, name="emb_mat")
            # query
            self.query_emb = tf.nn.embedding_lookup(self.embedding, self.query_in)
            self.query_emb_pool = tf.reduce_sum(self.query_emb, axis=1)
            # pos
            self.pos_emb = tf.nn.embedding_lookup(self.embedding, self.pos_in)
            self.pos_emb_pool = tf.reduce_sum(self.pos_emb, axis=1)
            # neg
            self.neg_emb = tf.nn.embedding_lookup(self.embedding, self.neg_in)
            self.neg_emb_pool = tf.reduce_sum(self.neg_emb, axis=1)
            # fake
            self.fake_emb = tf.nn.embedding_lookup(self.embedding, self.fake_in)
            self.fake_emb_pool = tf.reduce_sum(self.fake_emb, axis=1)

        with tf.variable_scope("query-fc", reuse=tf.AUTO_REUSE):
            self.query_vec = self.fc_layer(self.query_emb_pool, shape=[embedding_size, hidden_size],
                    name="query-fc", activation_function=tf.nn.tanh)
            # use predict query-query similarity
            self.fake_vec = self.fc_layer(self.fake_emb_pool, shape=[embedding_size, hidden_size],
                    name="query-fc", activation_function=tf.nn.tanh)
        with tf.variable_scope("pos-fc", reuse=tf.AUTO_REUSE):
            self.pos_vec = self.fc_layer(self.pos_emb_pool, shape=[embedding_size, hidden_size],
                    name="pos-fc", activation_function=tf.nn.tanh)
        with tf.variable_scope("neg-fc", reuse=tf.AUTO_REUSE):
            self.neg_vec = self.fc_layer(self.neg_emb_pool, shape=[embedding_size, hidden_size],
                    name="neg-fc", activation_function=tf.nn.tanh)

        with tf.variable_scope("consine", reuse=tf.AUTO_REUSE):
            self.qp_sim = cosine_similarity(self.query_vec, self.pos_vec)
            self.qn_sim = cosine_similarity(self.query_vec, self.neg_vec)
            self.diff_sim = tf.subtract(self.qp_sim, self.qn_sim)
            # use predict query-query similarity
            self.qq_sim = cosine_similarity(self.query_vec, self.fake_vec)

        with tf.variable_scope("loss"): 
            self.loss = rank_loss(self.qp_sim, self.qn_sim)

    def fc_layer(self, inputs, shape, name, activation_function=None):
        """ fc layer
        """
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            weights = tf.get_variable(name="%s_w" % name, shape=shape, 
                    initializer=tf.random_normal_initializer(mean=0, stddev=0.1))
            biases = tf.get_variable(name="%s_b" % name, shape=[shape[1]], 
                    initializer=tf.constant_initializer(0.001))
            wx_plus_b = tf.add(tf.matmul(inputs, weights), biases)
            if activation_function is None:
                outputs = wx_plus_b
            else:
                outputs = activation_function(wx_plus_b)
            return outputs

