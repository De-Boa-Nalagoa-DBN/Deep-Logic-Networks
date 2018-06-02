import numpy as np
from rbm import RBM
from rule import Rule
from modified_rbm import RBM2
import matplotlib.pyplot as plt
import tensorflow as tf
from utils import tile_raster_images
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data
from extract_knowledge import rbm_extract
from extract_knowledge import top_rbm_extract
from modified_dbn import DBN as DBN2

def dbn_extract(dbn, compact = True):
    ruleSet = []

    for i in range(len(dbn.w_list) -1):
        rbm = RBM2(dbn.w_list[i].shape[0], dbn.w_list[i].shape[1])
        rbm.load_weights(dbn.w_list[i])
        rules_rbm = rbm_extract(rbm.wUp)
        ruleSet.append(rules_rbm)
    
    if compact:
        rbm = RBM2(dbn.w_list[i].shape[0], dbn.w_list[i].shape[1])
        rbm.load_weights(dbn.w_list[-1])
        rules_rbm = rbm_extract(rbm.wUp)
        ruleSet.append(rules_rbm)
    else:
        rules_rbm = top_rbm_extract(dbn.w_list[-1])
        ruleSet.append(rules_rbm)
    return ruleSet

def main():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images,\
        mnist.test.labels

    dbn = DBN2([500, 200, 50], trX, trY, epochs=15)
    dbn.load_from_rbms([500, 200, 50], dbn.train_rbms())
    dbn.train()
    ruleSet = dbn_extract(dbn, compact=False)

    with open('rbm_rules','w') as rules_file:
        for i in range(len(ruleSet)):
            for r in ruleSet[i]:
                rules_file.write("{}, {} : {} <-> {}\n".format(i, r.c, r.h, r.x))
        rules_file.close()
if __name__ == '__main__':
    main()