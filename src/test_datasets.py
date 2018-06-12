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

def test_yale():
    dataset = []
    labels = []
    cont = 0

    for i in range(1, 16):
        filelist = glob.glob('./yale_dataset/subject'+str(i).zfill(2)+"*")
        for filename in filelist:
            cont += 1
            img = Image.open(filename).convert('L')
            img = img.resize((132, 132))
            img = np.array(img, dtype=np.float32)
            img = np.reshape(img, [img.shape[0]*img.shape[1]])
            img = img / 255.
            dataset.append(img)
            labels.append(float(i))

    dataset = np.asarray(dataset)
    labels = np.asarray(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        dataset, labels, test_size=31, random_state=42, stratify=labels)
    print(y_test)

    img_test = X_test[y_test == 11][0:1]
    # print(img_test)

    rbm = RBM(X_train.shape[1], 500)
    rbm.train(dataset, debug=True, epochs=500, batchsize=11, learning_rate=0.1)


    out = rbm.rbm_output(img_test, debug=True)
    v1 = rbm.backward_pass(out, rbm.w, rbm.vb)
    with tf.Session() as sess:
        feed = sess.run(out)
        out = sess.run(v1, feed_dict={out: feed})

    out = np.reshape(out, [132, 132])
    plt.imshow(out)
    plt.show()

    plt.imshow(np.reshape(img_test[0], (132, 132)))
    plt.show()

    # img = Image.fromarray(tile_raster_images(X=out, img_shape=(
    #     243, 320), tile_shape=(1, 1), tile_spacing=(1, 1)))
    # plt.rcParams['figure.figsize'] = (2.0, 2.0)
    # imgplot = plt.imshow(img)
    # imgplot.set_cmap('gray')
    # plt.show()

    input_ = tf.placeholder("float", [None, rbm._input_size])
    w_ = tf.placeholder("float", [rbm._input_size, rbm._output_size])
    hb_ = tf.placeholder("float", [rbm._output_size])
    with tf.Session() as sess:
        outputs_train = sess.run(rbm.foward_pass(input_, w_, hb_), feed_dict={
                            input_: X_train, w_: rbm.w, hb_: rbm.hb} )
        outputs_test = sess.run(rbm.foward_pass(input_, w_, hb_), feed_dict={
                            input_: X_test, w_: rbm.w, hb_: rbm.hb} )

    print(outputs_train.shape)
    print(outputs_test.shape)

    how_many = outputs_train.shape[0]
    svm_rbm = SVC()
    svm_rbm.fit(outputs_train, y_train)
    classi = svm_rbm.predict(outputs_test)
    som = 0
    for i,x in enumerate(y_test):
        print(classi[i])
        if x == classi[i]:
            som += 1
    print(som/X_test.shape[0])


def main():
    # test_yale()
    test_mnist(with_rules=True)

if __name__ == '__main__':
    main()