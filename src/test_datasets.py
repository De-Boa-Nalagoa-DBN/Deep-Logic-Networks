import numpy as np
import matplotlib.pyplot as plt
import glob
import tensorflow as tf
import extract_knowledge

from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from rbm import RBM
from utils import tile_raster_images
from inference import quantitativeInference


def test_mnist(with_rules=False, one_hot=False, hidden_units=500, n_test=1):
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=one_hot)
    trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images,\
        mnist.test.labels

    outputs = []
    how_many = 10000
    num_classes = 10
    accs = []
    wr_str = str(with_rules)
    hu_str = str(hidden_units)
    filename = "mnist_" + wr_str + "_" + hu_str + ".npy"

    for t in range(n_test):
        print("\n--------- TEST {} WITH{} RULES---------\n".format(t, "" if with_rules==True else "OUT"))
        rbm = RBM(trX.shape[1], hidden_units)
        rbm.train(trX, debug=False)
        print("!!! RBM trained !!!")

        if with_rules:
            rules_rbm = extract_knowledge.rbm_extract(rbm.w)
            print("Calculating beliefs...")
            new_trX = [quantitativeInference([rules_rbm], x) for x in trX]
            new_teX = [quantitativeInference([rules_rbm], x) for x in teX]
        else:
            input_ = tf.placeholder("float", [None, rbm._input_size])
            w_ = tf.placeholder("float", [rbm._input_size, rbm._output_size])
            hb_ = tf.placeholder("float", [rbm._output_size])
            with tf.Session() as sess:
                new_trX = sess.run(rbm.foward_pass(input_, w_, hb_), feed_dict={
                    input_: trX, w_: rbm.w, hb_: rbm.hb})
                new_teX = sess.run(rbm.foward_pass(input_, w_, hb_), feed_dict={
                    input_: teX, w_: rbm.w, hb_: rbm.hb})

        svm_inf = SVC()
        print("Fitting...")
        svm_inf.fit(new_trX, trY)
        print("Inferring...")
        pred = svm_inf.predict(new_teX)
        acc = (np.sum(pred == teY) / teY.shape[0]) * 100
        print("Accuracy: {}".format(acc))
        accs.append(acc)

        # Saving accuracy values
        accs_np = np.asarray(accs)
        np.save(filename, accs_np)

def test_yale(with_rules=False, hidden_units=500, n_test=1):
    dataset = []
    labels = []
    accs = []
    wr_str = str(with_rules)
    hu_str = str(hidden_units)
    filename = "yale_" + wr_str + "_" + hu_str + ".npy"

    # Preparing dataset
    for i in range(1, 16):
        filelist = glob.glob('./yale_dataset/subject'+str(i).zfill(2)+"*")
        for filename in filelist:
            img = Image.open(filename).convert('L')
            img = img.resize((132, 132))
            img = np.array(img, dtype=np.float32)
            img = np.reshape(img, [img.shape[0]*img.shape[1]])
            img = img / 255.
            dataset.append(img)
            labels.append(float(i))

    dataset = np.asarray(dataset)
    labels = np.asarray(labels)

    for t in range(n_test):
        X_train, X_test, y_train, y_test = train_test_split(
            dataset, labels, test_size=31, random_state=42, stratify=labels)

        print("\n--------- TEST {} WITH{} RULES---------\n".format(t, "" if with_rules==True else "OUT"))
        rbm = RBM(X_train.shape[1], with_rules)
        rbm.train(dataset, debug=True, epochs=500, batchsize=11, learning_rate=0.1)
        print("!!! RBM trained !!!")

        if with_rules:
            rules_rbm = extract_knowledge.rbm_extract(rbm.w)
            print("Calculating beliefs...")
            new_trX = [quantitativeInference([rules_rbm], x) for x in X_train]
            new_teX = [quantitativeInference([rules_rbm], x) for x in X_test]
        else:
            input_ = tf.placeholder("float", [None, rbm._input_size])
            w_ = tf.placeholder("float", [rbm._input_size, rbm._output_size])
            hb_ = tf.placeholder("float", [rbm._output_size])
            with tf.Session() as sess:
                new_trX = sess.run(rbm.foward_pass(input_, w_, hb_), feed_dict={
                                    input_: X_train, w_: rbm.w, hb_: rbm.hb} )
                new_teX = sess.run(rbm.foward_pass(input_, w_, hb_), feed_dict={
                                    input_: X_test, w_: rbm.w, hb_: rbm.hb} )

        svm_inf = SVC()
        print("Fitting...")
        svm_inf.fit(new_trX, y_train)
        print("Inferring...")
        pred = svm_inf.predict(new_teX)
        acc = (np.sum(pred == y_test) / y_test.shape[0]) * 100
        print("Accuracy: {}".format(acc))
        accs.append(acc)

        # Saving accuracy values
        accs_np = np.asarray(accs)
        np.save(filename, accs_np)

def main():
    # test_yale()
    test_mnist(with_rules=True)

if __name__ == '__main__':
    main()