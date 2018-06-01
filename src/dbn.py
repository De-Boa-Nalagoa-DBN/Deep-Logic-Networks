import math
import numpy as np
import tensorflow as tf
from PIL import Image
from rbm import RBM
from utils import tile_raster_images
from tensorflow.examples.tutorials.mnist import input_data


class DBN(object):

    def __init__(self, sizes, X, Y, epochs=10, batch_size=100, learning_rate=1.0, momentum=0.0):
        self._sizes = sizes
        self._X = X
        self._Y = Y
        # Creates weights and biases
        self.w_list = list()
        self.b_list = list()
        # train args
        self._learning_rate = learning_rate
        self._momentum = momentum
        self._epochs = epochs
        self._batch_size = batch_size
        input_size = X.shape[1]

        # initializing
        for size in self._sizes + [self._Y.shape[1]]:
            # Defining upper limit for the uniform distribution range
            max_range = 4 * math.sqrt(6. / (input_size + size))

            # Initialize weights through a random uniform distribution
            self.w_list.append(
                np.random.uniform(-max_range, max_range, [input_size, size]).astype(np.float32))

            # Initialize bias with zeros
            self.b_list.append(np.zeros([size], np.float32))
            input_size = size

    def train_rbms(self):
        inpX = self._X
        rbm_list = []
        input_size = inpX.shape[1]

        # for each RBM we want to generate
        for i, size in enumerate(self._sizes):
            rbm_list.append(RBM(input_size, size))
            input_size = size

        # for each RBM in our RBM's list
        for i, rbm in enumerate(rbm_list):
            print('\n\nRBM {}: '.format(i))
            rbm.train(inpX)
            inpX = rbm.rbm_output(inpX)
            rbm.save_weights(i)
            rbm.save_biases(i)

        return rbm_list

    def load_from_rbms(self, dbn_sizes, rbm_list):

        # Check if sizes are correct
        assert len(dbn_sizes) == len(
            self._sizes), "Sizes passed on load_from_rbms are wrong"
        for i in range(len(self._sizes)):
            assert dbn_sizes[i] == self._sizes[i]

        # Load weights and biases
        for i in range(len(self._sizes)):
            self.w_list[i] = rbm_list[i].w
            self.b_list[i] = rbm_list[i].hb

    def train(self):
        # Create placeholders for input, weeights, biases and output
        _in = [None] * (len(self._sizes) + 2)
        _w = [None] * (len(self._sizes) + 1)
        _b = [None] * (len(self._sizes) + 1)
        _in[0] = tf.placeholder("float", [None, self._X.shape[1]])
        y = tf.placeholder("float", [None, self._Y.shape[1]])

        # Initializing variables
        for i in range(len(self._sizes) + 1):
            _w[i] = tf.Variable(self.w_list[i])
            _b[i] = tf.Variable(self.b_list[i])

        # Defining activation function
        for i in range(1, len(self._sizes) + 2):
            _in[i] = tf.nn.sigmoid(
                tf.matmul(_in[i - 1], _w[i - 1]) + _b[i - 1])

        # Define cost function
        cost = tf.reduce_mean(tf.square(_in[-1] - y))

        # Defining that we want to minimize the cost function throught Tensor Flow's momentum optimizer
        train_op = tf.train.MomentumOptimizer(
            self._learning_rate, self._momentum).minimize(cost)

        # Prediction operation to fit later
        predict_op = tf.argmax(_in[-1], 1)

        # Effectly train
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(self._epochs):
                for start, end in zip(range(0, len(self._X), self._batch_size), range(self._batch_size, len(self._X), self._batch_size)):
                    # Fits current batch on dbn
                    sess.run(train_op, feed_dict={
                             _in[0]: self._X[start:end], y: self._Y[start:end]})

                for j in range(len(self._sizes) + 1):
                    # Updates weights and biases
                    self.w_list[j] = sess.run(_w[j])
                    self.b_list[j] = sess.run(_b[j])

                print("Accuracy rating for epoch " + str(i) + ": " + str(np.mean(np.argmax(self._Y, axis=1) ==
                                                                                 sess.run(predict_op, feed_dict={_in[0]: self._X, y: self._Y}))))

    def predict(self, X):
        # Create placeholders for input, weeights, biases and output
        _in = [None] * (len(self._sizes) + 2)
        _w = [None] * len(self.w_list)
        _b = [None] * len(self.b_list)
        _in[0] = tf.placeholder("float", [None, self._X.shape[1]])

        # Initializing variables
        for i in range(len(self.w_list)):
            _w[i] = tf.constant(self.w_list[i])
            _b[i] = tf.constant(self.b_list[i])

        # Defining activation function
        for i in range(1, len(self._sizes) + 2):
            _in[i] = tf.nn.sigmoid(
                tf.matmul(_in[i - 1], _w[i - 1]) + _b[i - 1])

        # Prediction operation
        predict_op = tf.argmax(_in[-1], 1)

        # predict
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            return sess.run(predict_op, feed_dict={_in[0]: X})


def main():
    # Loading in the mnist data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images,\
        mnist.test.labels

    dbn = DBN([500, 200, 50], trX, trY, epochs=15)
    dbn.load_from_rbms([500, 200, 50], dbn.train_rbms())
    dbn.train()
    print("Prediction for Mnist dataset with DBN without learning extraction and insertion:",
          np.mean(np.argmax(teY, axis=1) == dbn.predict(teX)))


if __name__ == '__main__':
    main()
