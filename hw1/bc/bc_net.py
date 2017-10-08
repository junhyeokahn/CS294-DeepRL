#!/usr/bin/env python

import tensorflow as tf

class BehaviorCloneNetwork(object):
    def __init__(self, optimizer, lr):
        self.obs = tf.placeholder(dtype=tf.float32, shape=[None, 11], name="Obs")
        self.actions = tf.placeholder(dtype=tf.float32, shape=[None, 3], name="Actions")
        self.pred, self.parameters = self.network()
        self.loss = self.get_loss()
        self.optimizer = self.get_optimizer(optimizer, lr)
        self.saver = tf.train.Saver(var_list=tf.global_variables())

    def get_loss(self):
        """ return loss """
        loss = tf.reduce_mean(tf.pow((self.pred - self.actions), 2)) / 2
        # loss = tf.nn.l2_loss(self.pred - self.actions)
        return loss

    def get_optimizer(self, optimizer, lr):
        """ initialize optimizer """
        print("[Optimizer %s is Initialized]" % optimizer)
        if optimizer == "adam":
            return tf.train.AdamOptimizer(lr).minimize(self.loss)
        elif optimizer == "adagrad":
            return tf.train.AdagradOptimizer(lr).minimize(self.loss)
        else:
            return tf.train.GradientDescentOptimizer(lr).minimize(self.loss)

    def network(self):
        """ connect with two hidden layer """

        parameters = []
        with tf.variable_scope("h1"):
            # w_h1 = tf.get_variable('Weight', [11, 128],
            w_h1 = tf.get_variable('Weight', [11, 64],
                    initializer=tf.contrib.layers.xavier_initializer(),
                    dtype=tf.float32)
            # b_h1 = tf.get_variable('Bais', [128],
            b_h1 = tf.get_variable('Bais', [64],
                    initializer=tf.constant_initializer(0),
                    dtype=tf.float32)
            z_h1 = tf.add(tf.matmul(self.obs, w_h1), b_h1)
            a_h1 = tf.nn.relu(z_h1)
            parameters += [w_h1, b_h1]
        with tf.variable_scope("h2"):
            # w_h2 = tf.get_variable('Weight', [128, 32],
            w_h2 = tf.get_variable('Weight', [64, 16],
                    initializer=tf.contrib.layers.xavier_initializer(),
                    dtype=tf.float32)
            # b_h2 = tf.get_variable('Bais', [32],
            b_h2 = tf.get_variable('Bais', [16],
                    initializer=tf.constant_initializer(0),
                    dtype=tf.float32)
            z_h2 = tf.add(tf.matmul(a_h1, w_h2), b_h2)
            a_h2 = tf.nn.relu(z_h2)
            parameters += [w_h2, b_h2]
        with tf.variable_scope("output"):
            # w_ol = tf.get_variable('Weight', [32, 3],
            w_ol = tf.get_variable('Weight', [16, 3],
                    initializer=tf.contrib.layers.xavier_initializer(),
                    dtype=tf.float32)
            b_ol = tf.get_variable('Bais', [3],
                    initializer=tf.constant_initializer(0),
                    dtype=tf.float32)
            z_ol = tf.add(tf.matmul(a_h2, w_ol), b_ol)
            # a_ol = tf.nn.relu(z_ol)
            parameters += [w_ol, b_ol]

        return z_ol, parameters

    def step(self, sess, batch_x, batch_y=None, is_train=True):
        feed_dict = {self.obs: batch_x}
        if batch_y is not None:
            feed_dict[self.actions] = batch_y
        if is_train:
            _, pred, loss = sess.run([self.optimizer, self.pred, self.loss],
                    feed_dict=feed_dict)
            return pred, loss
        else:
            if batch_y is not None:
                pred, loss = sess.run([self.pred, self.loss], feed_dict=feed_dict)
                return pred, loss
            else:
                pred = sess.run([self.pred], feed_dict=feed_dict)
                return pred
