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
