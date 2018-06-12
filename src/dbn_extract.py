import numpy as np
from rbm import RBM
from rule import Rule
from modified_rbm import RBM2
import tensorflow as tf
from utils import tile_raster_images
from tensorflow.examples.tutorials.mnist import input_data
from extract_knowledge import rbm_extract
from modified_dbn import DBN as DBN2
import extract_dbn

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images,\
        mnist.test.labels

dbn = DBN2([500, 1000], trX, trY, teX, teY, epochs=500)
dbn.load_from_rbms([500, 1000], dbn.train_rbms())
dbn.train()

knowledgeBase = extract_dbn.dbn_extract(dbn, compact=False)

sizes = [len(knowledgeBase[i]) for i in range(len(knowledgeBase))]
dbn = DBN2(sizes, trX, trY, teX, teY, epochs=500, learning_rate=0.05)
dbn.load_from_rbms2(sizes, dbn.ruleEncodingAlgorithm(knowledgeBase))
dbn.train()