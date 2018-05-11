import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data

from utils import tile_raster_images


class RBM:

    _h0 = _v1 = h0 = v1 = h1 = update_w = update_vb = update_hb = None

    # initializating visible and hidden biases, weights and the input matrixes
    def __init__(self):
        self.vb = tf.placeholder("float", [784])
        self.hb = tf.placeholder("float", [500])
        self.W = tf.placeholder('float', [784, 500])
        self.X = tf.placeholder('float', [None, 784])
        self.cur_w = np.zeros([784, 500], np.float32)
        self.cur_vb = np.zeros([784], np.float32)
        self.cur_hb = np.zeros([500], np.float32)
        self.prev_w = np.zeros([784, 500], np.float32)
        self.prev_vb = np.zeros([784], np.float32)
        self.prev_hb = np.zeros([500], np.float32)

    def foward_pass(self):
        # hidden units' probabilities
        self._h0 = tf.nn.sigmoid(tf.matmul(self.X, self.W) + self.hb)
        # sample h given an input
        self.h0 = tf.nn.relu(
            tf.sign(self._h0 + tf.random_uniform(tf.shape(self._h0))))

    def backward_pass(self):
        # visible units' probabilities
        self._v1 = tf.nn.sigmoid(
            tf.matmul(self.h0, tf.transpose(self.W)) + self.vb)
        # sample v given a h
        self.v1 = tf.nn.relu(
            tf.sign(self._v1 - tf.random_uniform(tf.shape(self._v1))))
        # resample the hidden activation
        self.h1 = tf.nn.sigmoid(tf.matmul(self.v1, self.W) + self.hb)

    def constrative_divergence(self, epochs=5, batch_size=100, train_data=None):
        alpha = 0.995

        self.foward_pass()
        # reconstruct the foward pass
        w_positive_gradient = tf.matmul(tf.transpose(self.X), self.h0)
        self.backward_pass()
        # reconstruct the backward pass
        w_negative_gradient = tf.matmul(tf.transpose(self.v1), self.h1)
        # calculates the new constrative divergence
        CD = (w_positive_gradient - w_negative_gradient) / \
            tf.to_float(tf.shape(self.X)[0])
        # update Weights and visible and hidden biases
        self.update_w = self.W + alpha * CD
        self.update_vb = self.vb + alpha * tf.reduce_mean(self.X - self.v1, 0)
        self.update_hb = self.hb + alpha * tf.reduce_mean(self.h0 - self.h1, 0)

    def train(self, epochs=5, batchsize=100, data_train=None):
        err = tf.reduce_mean(tf.square(self.X - self.v1))
        weights = []
        errors = []
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)

        for epoch in range(epochs):
            self.constrative_divergence()
            for start, end in zip(range(0, len(data_train), batchsize), range(batchsize, len(data_train), batchsize)):
                batch = data_train[start:end]
                self.cur_w = sess.run(self.update_w, feed_dict={
                                      self.X: batch, self.W: self.prev_w, self.vb: self.prev_vb, self.hb: self.prev_hb})
                self.cur_hb = sess.run(self.update_hb, feed_dict={
                                       self.X: batch, self.W: self.prev_w, self.vb: self.prev_vb, self.hb: self.prev_hb})
                self.cur_vb = sess.run(self.update_vb, feed_dict={
                                       self.X: batch, self.W: self.prev_w, self.vb: self.prev_vb, self.hb: self.prev_hb})
                self.prev_w = self.cur_w
                self.prev_hb = self.cur_hb
                self.prev_vb = self.cur_vb

                if start % 10000 == 0:
                    errors.append(sess.run(err, feed_dict={
                                  self.X: data_train, self.W: self.cur_w, self.vb: self.cur_vb, self.hb: self.cur_hb}))
                    weights.append(self.cur_w)
            print('Epoch: %d' % epoch, 'reconstruction error: %f' % errors[-1])

        plt.plot(errors)
        plt.xlabel("Batch Number")
        plt.ylabel("Error")
        plt.show()

def main():
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
    train_X, train_Y, test_X, test_Y = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels


if __name__ == '__main__':
    main()
