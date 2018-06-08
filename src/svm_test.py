import numpy as np
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
    rules_rbm = extract_knowledge.rbm_extract(rbm.w)
    confidence_values = [x.c for x in rules_rbm]
    confidence_values = np.asarray(
        confidence_values).reshape(len(confidence_values), 1)
    svm_rbm = svm.SVC()
    svm_confidence = svm.SVC()
    svm_rbm.fit(rbm.hb, trY)
    svm_confidence.fit(confidence_values, trY)
    acc_rbm = accuracy_score(teY,svm_rbm.predict(teX))
    acc_confidence = accuracy_score(teY,svm_confidence.predict(teX))
    print(acc_rbm, acc_confidence)


if __name__ == '__main__':
    main()
