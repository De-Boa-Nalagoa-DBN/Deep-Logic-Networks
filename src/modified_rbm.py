import math

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#from PIL import Image

from utils import tile_raster_images

HIDDEN_UNITS    = [500, 1000]
SAVE_DIR        = "trained/" 

class RBM2:

    # initializating visible and hidden biases, weights and the input matrixes
    def __init__(self, input_size, output_size):
        self._input_size = input_size
        self._output_size = output_size
        # Initializing weights and biases as matrices full of zeroes
        self.wUp = np.zeros([input_size, output_size], np.float32)
        self.wDown = np.zeros([output_size, input_size], np.float32)
        self.hb = np.zeros([output_size], np.float32)
        self.vb = np.zeros([input_size], np.float32)
        self.multiplier = np.ones([input_size, output_size], np.float32)

    def load_weights(self, weights):
       pass

    def load_biases(self, dir_vb, dir_hb):
        print("Loading visible biases from {} ...".format(dir_vb))
        print("Loading hidden biases from {} ...".format(dir_hb))
        tmp_vb = np.load(dir_vb)
        tmp_hb = np.load(dir_hb)
        if tmp_vb.shape != self.vb.shape:
            print("Error: Loaded visible biases shape: {}, expected shape: {}!".format(tmp_vb.shape, self.vb.shape))
            return -1
        if tmp_hb.shape != self.hb.shape:
            print("Error: Loaded hidden biases shape: {}, expected shape: {}!".format(tmp_hb.shape, self.hb.shape))
            return -1

        self.vb = tmp_vb
        self.hb = tmp_hb
        print("Biases loaded successfully!")

    def save_weights(self):
        print("Saving weights in {} ...".format(SAVE_DIR))
        np.save(SAVE_DIR + "rbm_weightsDown.npy", self.wDown)
        np.save(SAVE_DIR + "rbm_weightsUp.npy", self.wUp)

    def save_biases(self):
        print("Saving biases in {} ...".format(SAVE_DIR))
        np.save(SAVE_DIR + "rbm_vb.npy", self.vb)
        np.save(SAVE_DIR + "rbm_hb.npy", self.hb)

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
        _wDown = tf.placeholder("float", [self._input_size, self._output_size])
        _wUp = tf.placeholder("float", [self._input_size, self._output_size])
        _hb = tf.placeholder("float", [self._output_size])
        _vb = tf.placeholder("float", [self._input_size])
        _multiplier = tf.placeholder("float", [self._input_size, self._output_size])
        v0 = tf.placeholder("float", [None, self._input_size])
        

        # initialize hidden and visible biases with 0
        cur_wDown = np.zeros([self._input_size, self._output_size], np.float32)
        cur_wUp = np.zeros([self._input_size, self._output_size], np.float32)
        cur_vb = np.zeros([self._input_size], np.float32)
        cur_hb = np.zeros([self._output_size], np.float32)

        prev_wUp = self.wUp
        prev_wDown = np.zeros([self._input_size, self._output_size], np.float32)
        prev_vb = np.zeros([self._input_size], np.float32)
        prev_hb = np.zeros([self._output_size], np.float32)

        # Sample the probabilities
        h0 = self.sample_prob(self.foward_pass(v0, _wUp, _hb))
        v1 = self.sample_prob(self.backward_pass(h0, _wDown, _vb))
        h1 = self.foward_pass(v1, _wUp, _hb)

        # Do Contrastive Divergence:
        # Calculates gradients
        positive_gradient = tf.matmul(tf.transpose(v0), h0)
        negative_gradient = tf.matmul(tf.transpose(v1), h1)

        alpha = learning_rate

        # calculates the new constrative divergence
        CD = (positive_gradient - negative_gradient) / \
            tf.to_float(tf.shape(v0)[0])
        # update Weights and visible and hidden biases
        update_w = _wDown + alpha * CD
        update_wUp = _wUp + alpha *np.multiply(CD, _multiplier) 
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
                    cur_wDown = sess.run(update_w, feed_dict={
                        v0: batch, _wDown: prev_wDown, _wUp: prev_wUp, _vb: prev_vb, _hb: prev_hb})
                    cur_wUp = sess.run(update_wUp, feed_dict={
                        v0: batch, _wDown: prev_wDown, _wUp: prev_wUp, _vb: prev_vb, _hb: prev_hb, _multiplier: self.multiplier})    
                    cur_hb = sess.run(update_hb, feed_dict={
                        v0: batch, _wDown: prev_wDown, _wUp: prev_wUp, _vb: prev_vb, _hb: prev_hb})
                    cur_vb = sess.run(update_vb, feed_dict={
                        v0: batch, _wUp: prev_wUp,_wDown: prev_wDown, _vb: prev_vb, _hb: prev_hb})
                    prev_wDown = cur_wDown
                    prev_wUp = cur_wUp
                    prev_hb = cur_hb
                    prev_vb = cur_vb

                    if start % 10000 == 0 and debug:
                        error = (sess.run(err, feed_dict={
                            v0: data_train, _wDown: cur_wDown, _wUp: prev_wUp, _vb: cur_vb, _hb: cur_hb}))
                
                if debug:
                    print('Epoch: {}, reconstruction error: {}'.format(epoch, error))

                self.wDown = prev_wDown
                self.wUp = prev_wUp
                self.hb = prev_hb
                self.vb = prev_vb

    def rbm_output(self, X):
        input_X = tf.constant(X)
        _w = tf.constant(self.wDown)
        _hb = tf.constant(self.hb)
        out = tf.nn.sigmoid(tf.matmul(input_X, _w) + _hb)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            return sess.run(out)
    
    def insertKnowledge(self, ruleSet):
        for i in range(len(ruleSet)):
            for j in range(len(ruleSet[i].x)):
                if ruleSet[i].x[j] == True:
                    self.multiplier[j][i] = 0.0
                    self.wUp[j][i] = ruleSet[i].c
                elif ruleSet[i].x[j] == False:
                    self.multiplier[j][i] = 0.0
                    self.wUp[j][i] = -ruleSet[i].c

def main():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images,\
        mnist.test.labels

    rbm = RBM2(trX.shape[1], HIDDEN_UNITS[0])
    rbm.train(trX, debug=True)

    # Saving weights and biases
    rbm.save_weights()
    rbm.save_biases()

if __name__ == '__main__':
    main()