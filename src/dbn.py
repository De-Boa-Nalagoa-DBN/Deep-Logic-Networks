import math
import numpy as np
import tensorflow as tf
from PIL import Image
from rbm import RBM
from utils import tile_raster_images
from tensorflow.examples.tutorials.mnist import input_data






def main():
    #Loading in the mnist data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images,\
        mnist.test.labels

if __name__ == '__main__':
    main()