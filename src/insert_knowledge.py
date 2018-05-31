import numpy as np
from rbm import RBM
from dbn import DBN
import tensorflow as tf
from rule import Rule

# knowledgeBase [[Rule]]
def ruleEncodingAlgorithm(knowledgeBase):
    network = []

    for l in len(knowledgeBase):
        current = RBM( len(knowledgeBase[l][0][0].x), len(knowledgeBase[l][0]))

        for i in range(len(knowledgeBase[l])):

            for j in range(len(knowledgeBase[l][i].x)):

                if knowledgeBase[l][i].x[j] == True:
                    current.w[i][j] = knowledgeBase[l][i].c
                elif knowledgeBase[l][i].x[j] == False:
                    current.w[i][j] = -knowledgeBase[l][i].c
        
        network.append(current)
    
    dbn = DBN([network[0]._input_size] + [x._output_size for x in network], [],[])

    return dbn.load_from_rbms(len(network) - 1, network)
            

def learningWithGuidance(knowledgeBase, X, Y):
    dbn = ruleEncodingAlgorithm(knowledgeBase)

    dbn.train_with_guidance(X,Y)
    
    return dbn

