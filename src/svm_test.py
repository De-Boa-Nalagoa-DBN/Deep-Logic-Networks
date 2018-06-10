import numpy as np
import tensorflow as tf
from sklearn import svm
from sklearn.metrics import accuracy_score
from tensorflow.examples.tutorials.mnist import input_data

import extract_knowledge
from rbm import RBM


def main():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images,\
        mnist.test.labels

    rbm = RBM(trX.shape[1], 500)
    rbm.train(trX, debug=False)
    outputs = []
    for i in range(200):
        out = rbm.rbm_output(trX[i:i+1], True)
        v1 = rbm.backward_pass(out, rbm.w, rbm.vb)
        with tf.Session() as sess:
            feed = sess.run(out)
            out = sess.run(v1, feed_dict={out: feed})
        outputs.append(out[0])

    outputs = np.asarray(outputs)



    rules_rbm = extract_knowledge.rbm_extract(rbm.w)
    confidence_values = [x.c for x in rules_rbm]
    confidence_values = np.asarray(
        confidence_values).reshape(len(confidence_values), 1)
    rbm_conf = rbm
    rbm_conf.hb = confidence_values

    # rbm.rbm_output(teX)
    svm_rbm = svm.SVC()
    y = []
    for x in trY[:200]:
        for i, l in enumerate(x):
            if l == 1:
                y.append(i)
    svm_rbm.fit(outputs, y)
    classi = svm_rbm.predict(teX[:200])
    print(classi)
    print(teY[:10])
    som = 0
    for i,x in enumerate(teY[:200]):
        x = [j for j in range(len(x)) if x[j]==1]
        if x[0] == classi[i]:
            som+= 1
    print(som/200)
    # svm_confidence = svm.SVC()

    # svm_rbm.fit(rbm.hb, trY)
    # svm_confidence.fit(confidence_values, trY)
    # acc_rbm = accuracy_score(teY,svm_rbm.predict(teX))
    # acc_confidence = accuracy_score(teY,svm_confidence.predict(teX))
    # print(acc_rbm, acc_confidence)


if __name__ == '__main__':
    main()
