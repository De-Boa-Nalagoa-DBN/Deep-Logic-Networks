import numpy as np
import tensorflow as tf
from sklearn import svm
from sklearn.metrics import accuracy_score
from tensorflow.examples.tutorials.mnist import input_data

import extract_knowledge
from inference import quantitativeInference
from rbm import RBM


def main():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images,\
        mnist.test.labels

    rbm = RBM(trX.shape[1], 500)
    rbm.train(trX, debug=False)
    outputs = []
    how_many = 10000

    input_ = tf.placeholder("float", [None, rbm._input_size])
    w_ = tf.placeholder("float", [rbm._input_size, rbm._output_size])
    hb_ = tf.placeholder("float", [rbm._output_size])
    with tf.Session() as sess:
        outputs = sess.run(rbm.foward_pass(input_, w_, hb_), feed_dict={
            input_: trX, w_: rbm.w, hb_: rbm.hb})
        outputs_test = sess.run(rbm.foward_pass(input_, w_, hb_), feed_dict={
            input_: teX, w_: rbm.w, hb_: rbm.hb})

    rules_rbm = extract_knowledge.rbm_extract(rbm.w)
    print('Aqui1')
    newTrX = [quantitativeInference([rules_rbm], x) for x in trX]
    print('Aqui2')
    newTeX = [quantitativeInference([rules_rbm], x) for x in teX]
    print('Aqui3')

    # svm_rbm = svm.SVC()
    # y = []
    # for x in trY:
    #     for i, l in enumerate(x):
    #         if l == 1:
    #             y.append(i)
    # svm_rbm.fit(outputs, y)
    # print("Fitou")
    # classi = svm_rbm.predict(outputs_test)
    # print(classi)
    # som = 0
    # for i,x in enumerate(teY):
    #     x = [j for j in range(len(x)) if x[j]==1]
    #     if x[0] == classi[i]:
    #         som+= 1
    # print(som/how_many)

    svm_inf = svm.SVC()
    y = []
    for x in trY:
        for i, l in enumerate(x):
            if l == 1:
                y.append(i)
    svm_inf.fit(newTrX, y)
    print('Fitou')
    classi = svm_inf.predict(newTeX)
    som = 0
    for i, x in enumerate(teY):
        x = [j for j in range(len(x)) if x[j] == 1]
        if x[0] == classi[i]:
            som += 1

    print("Final accuracy:", som/how_many)


if __name__ == '__main__':
    main()
