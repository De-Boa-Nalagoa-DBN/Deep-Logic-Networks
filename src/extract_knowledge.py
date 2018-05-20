import numpy as np
from rbm import RBM
from rule import Rule

def get_condifence(w, s, j):
    num = np.sum(np.absolute(w.T[j]))
    den = np.sum(np.power(s.T[j], 2))
    c = num / den

    return c

def rbm_extract(W):
    n_visible = W.shape[0]
    n_hidden = W.shape[1]
    S = np.zeros(W.shape)
    r = []

    for j in range(n_hidden):
        print("Calculating confidence for hidden unit: {}".format(j))
        cj = None
        r.append(Rule(j))
        for i in range(n_visible):
            S[i][j] = np.sign(W[i][j])
            r[j].add_literal(W[i][j])

        print(r[j].x)

        while cj != r[j].c:
            r[j].c = cj
            cj = get_condifence(W, S, j)
            for i in range(n_visible):
                if cj >= (2*np.absolute(W[i][j])) and S[i][j] != 0:
                    S[i][j] = 0
                    r[j].remove_literal(i)

        r[j].c = cj

    for rj in r:
        print("{} : {} <-> {}".format(rj.c, rj.h, rj.x))

def main():
    # TODO
    print("Extract Knowledge")
    W = np.load("trained/rbm_weights.npy")
    rbm_extract(W)

if __name__ == "__main__":
    main()