from rbm import RBM
from inf1 import inferenceRule

def quantInference(layers):
    b = l.w #nao sei se isso esta correto, acho que nao
    for l in len(layers):
        current = RBM(len(layers[l][0][0]), len(layers[l][0]))
        next = RBM(len(layers[l + 1][0][0]), len(layers[l + 1][0]))
        for rule in l:
            alfa = inferenceRule(rule, 0, 1, current.x)
            #alfa = normalize(alfa)
            #TODO COMO VCS ESTAO NORMALIZANDO?
            next.c += alfa
        current = next