import numpy as np
import matplotlib.pyplot as plt
import glob
import tensorflow as tf

from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from rbm import RBM
from utils import tile_raster_images


def test_yale():
    dataset = []
    labels = []

    for i in range(1, 16):
        filelist = glob.glob('./yale_dataset/subject'+str(i).zfill(2)+"*")
        for filename in filelist:
            img = Image.open(filename).convert('L')
            img = img.resize((28, 28))
            img = np.array(img, dtype=np.float32)
            img = np.reshape(img, [img.shape[0]*img.shape[1]])
            img = img / 255.
            dataset.append(img)
            labels.append(float(i))

    dataset = np.asarray(dataset)
    labels = np.asarray(labels)
    
    X_train, X_test, y_train, y_test = train_test_split(
        dataset, labels, test_size=31, random_state=42)

    rbm = RBM(X_train.shape[1], 500)
    rbm.train(X_train, debug=True, epochs=100, batchsize=1, learning_rate=0.1)

    out = rbm.rbm_output(X_train[0:1], debug=True)


    v1 = rbm.backward_pass(out, rbm.w, rbm.vb)
    with tf.Session() as sess:
        feed = sess.run(out)
        out = sess.run(v1, feed_dict={out: feed})

    out = np.reshape(out, [28, 28])
    plt.imshow(out)
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
    test_yale()

if __name__ == '__main__':
    main()