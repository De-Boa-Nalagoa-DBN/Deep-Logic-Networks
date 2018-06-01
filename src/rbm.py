import math

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image

from utils import tile_raster_images

HIDDEN_UNITS = [500, 1000]
SAVE_DIR = "trained/"


class RBM:

    # initializating visible and hidden biases, weights and the input matrixes
    def __init__(self, input_size, output_size):
        self._input_size = input_size
        self._output_size = output_size
        # Initializing weights and biases as matrices full of zeroes
        self.w = np.zeros([input_size, output_size], np.float32)
        self.hb = np.zeros([output_size], np.float32)
        self.vb = np.zeros([input_size], np.float32)

    def load_weights(self, dir):
        print("Loading weights from {} ...".format(dir))
        tmp_w = np.load(dir)
        if tmp_w.shape != self.w.shape:
            print("Error: Loaded weights shape: {}, expected shape: {}!".format(
                tmp_w.shape, self.w.shape))
            return -1
        else:
            self.w = tmp_w
            print("Weights loaded successfully!")

    def load_biases(self, dir_vb, dir_hb):
        print("Loading visible biases from {} ...".format(dir_vb))
        print("Loading hidden biases from {} ...".format(dir_hb))
        tmp_vb = np.load(dir_vb)
        tmp_hb = np.load(dir_hb)
        if tmp_vb.shape != self.vb.shape:
            print("Error: Loaded visible biases shape: {}, expected shape: {}!".format(
                tmp_vb.shape, self.vb.shape))
            return -1
        if tmp_hb.shape != self.hb.shape:
            print("Error: Loaded hidden biases shape: {}, expected shape: {}!".format(
                tmp_hb.shape, self.hb.shape))
            return -1

        self.vb = tmp_vb
        self.hb = tmp_hb
        print("Biases loaded successfully!")

    def save_weights(self, layer_n):
        print("Saving weights in {} ...".format(SAVE_DIR))
        np.save(SAVE_DIR + "rbm_weights_" + str(layer_n) +".npy", self.w)

    def save_biases(self, layer_n):
        print("Saving biases in {} ...".format(SAVE_DIR))
        np.save(SAVE_DIR + "rbm_vb_" + str(layer_n) +".npy", self.vb)
        np.save(SAVE_DIR + "rbm_hb_" + str(layer_n) +".npy", self.hb)

    def foward_pass(self, visible, w, hb):
        # hidden units' probabilities
        return tf.nn.sigmoid(tf.matmul(visible, w) + hb)

    def sample_prob(self, probs):
        # sample h given an input
        return tf.nn.relu(
            tf.sign(probs - tf.random_uniform(tf.shape(probs))))

    def backward_pass(self, hidden, w, vb):
        # visible units' probabilities
        return tf.nn.sigmoid(
            tf.matmul(hidden, tf.transpose(w)) + vb)

    def train(self, data_train=None, epochs=5, batchsize=100, learning_rate=1.0, debug=False):
        # Create the placeholders for our parameters
        _w = tf.placeholder("float", [self._input_size, self._output_size])
        _hb = tf.placeholder("float", [self._output_size])
        _vb = tf.placeholder("float", [self._input_size])
        v0 = tf.placeholder("float", [None, self._input_size])

        # initialize hidden and visible biases with 0

        cur_w = np.random.rand(self._input_size, self._output_size)
        cur_vb = np.random.rand(self._input_size)
        cur_hb = np.random.rand(self._output_size)

        prev_w = np.random.rand(self._input_size, self._output_size)
        prev_vb = np.random.rand(self._input_size)
        prev_hb = np.random.rand(self._output_size)

        # Sample the probabilities
        h0 = self.sample_prob(self.foward_pass(v0, _w, _hb))
        v1 = self.sample_prob(self.backward_pass(h0, _w, _vb))
        h1 = self.foward_pass(v1, _w, _hb)

        # Do Contrastive Divergence:
        # Calculates gradients
        positive_gradient = tf.matmul(tf.transpose(v0), h0)
        negative_gradient = tf.matmul(tf.transpose(v1), h1)

        alpha = learning_rate

        # calculates the new constrative divergence
        CD = (positive_gradient - negative_gradient) / \
            tf.to_float(tf.shape(v0)[0])
        # update Weights and visible and hidden biases
        update_w = _w + alpha * CD
        update_vb = _vb + alpha * tf.reduce_mean(v0 - v1, 0)
        update_hb = _hb + alpha * tf.reduce_mean(h0 - h1, 0)

        # calculates error rate
        err = tf.reduce_mean(tf.square(v0 - v1))

        # Train
        with tf.Session() as sess:

            init = tf.global_variables_initializer()
            sess.run(init)
            error = None

            for epoch in range(epochs):
                for start, end in zip(range(0, len(data_train), batchsize), range(batchsize, len(data_train), batchsize)):
                    batch = data_train[start:end]
                    # update weight and biases
                    cur_w = sess.run(update_w, feed_dict={
                        v0: batch, _w: prev_w, _vb: prev_vb, _hb: prev_hb})
                    cur_hb = sess.run(update_hb, feed_dict={
                        v0: batch, _w: prev_w, _vb: prev_vb, _hb: prev_hb})
                    cur_vb = sess.run(update_vb, feed_dict={
                        v0: batch, _w: prev_w, _vb: prev_vb, _hb: prev_hb})
                    prev_w = cur_w
                    prev_hb = cur_hb
                    prev_vb = cur_vb

                    if start % 10000 == 0 and debug:
                        error = (sess.run(err, feed_dict={
                            v0: data_train, _w: cur_w, _vb: cur_vb, _hb: cur_hb}))

                if debug:
                    print('Epoch: {}, reconstruction error: {}'.format(epoch, error))

                self.w = prev_w
                self.hb = prev_hb
                self.vb = prev_vb

    def rbm_output(self, X, debug=False):
        input_X = tf.constant(X)
        _w = tf.constant(self.w)
        _hb = tf.constant(self.hb)
        out = self.foward_pass(input_X, _w, _hb)
        if not debug:
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                return sess.run(out)
        else:
            return out


def main():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images,\
        mnist.test.labels
    
    img = Image.fromarray(tile_raster_images(X=trX[1:2], img_shape=(
        28, 28), tile_shape=(1, 1), tile_spacing=(1, 1)))
    plt.rcParams['figure.figsize'] = (2.0, 2.0)
    imgplot = plt.imshow(img)
    imgplot.set_cmap('gray')
    plt.show()
    # trX = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]]
    # trX = np.asarray(trX)
    # print(trX.shape[1])
    rbm = RBM(trX.shape[1], 500)
    rbm.train(trX, debug=True)

    # Saving weights and biases
    rbm.save_weights(0)
    rbm.save_biases(0)

    out = rbm.rbm_output(trX[1:2], debug=True)


    v1 = rbm.backward_pass(out, rbm.w, rbm.vb)
    with tf.Session() as sess:
        feed = sess.run(out)
        out = sess.run(v1, feed_dict={out: feed})
    img = Image.fromarray(tile_raster_images(X=out, img_shape=(
        28, 28), tile_shape=(1, 1), tile_spacing=(1, 1)))
    plt.rcParams['figure.figsize'] = (2.0, 2.0)
    imgplot = plt.imshow(img)
    imgplot.set_cmap('gray')
    plt.show()


if __name__ == '__main__':
    main()
