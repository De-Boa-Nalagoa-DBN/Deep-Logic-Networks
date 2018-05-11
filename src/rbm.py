import numpy as np
import tensorflow as tf


class RBM:

    # initializating visible and hidden biases, weights and the input matrixes
    def __init__(self, input_size, output_size):
        self._input_size = input_size
        self._output_size = output_size
        # Initializing weights and biases as matrices full of zeroes
        self.W = np.zeros([input_size, output_size], np.float32)
        self.hb = np.zeros([output_size], np.float32)
        self.vb = np.zeros([input_size], np.float32)

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

    def train(self, data_train=None, epochs=5, batchsize=100, learning_rate=1.0):
        # Create the placeholders for our parameters
        _w = tf.placeholder("float", [self._input_size, self._output_size])
        _hb = tf.placeholder("float", [self._output_size])
        _vb = tf.placeholder("float", [self._input_size])
        v0 = tf.placeholder("float", [None, self._input_size])

        # initialize hidden and visible biases with 0
        cur_w = np.zeros([self._input_size, self._output_size], np.float32)
        cur_vb = np.zeros([self._input_size], np.float32)
        cur_hb = np.zeros([self._output_size], np.float32)

        prev_w = np.zeros([self._input_size, self._output_size], np.float32)
        prev_vb = np.zeros([self._input_size], np.float32)
        prev_hb = np.zeros([self._output_size], np.float32)

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
                self.w = prev_w
                self.hb = prev_hb
                self.vb = prev_vb
                #while testing:
                error = (sess.run(err, feed_dict={
                    v0: data_train, _w: cur_w, _vb: cur_vb, _hb: cur_hb}))
                print('Epoch: %d' % epoch, 'reconstruction error: %f' % error)

    def rbm_output(self, X):
        input_X = tf.constant(X)
        _w = tf.constant(self.w)
        _hb = tf.constant(self.hb)
        out = tf.nn.sigmoid(tf.matmul(input_X, _w) + _hb)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            return sess.run(out)
