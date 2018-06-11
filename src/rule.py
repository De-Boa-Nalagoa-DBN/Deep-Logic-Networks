import numpy as np

class Rule:
    def __init__(self, h):
        self.c = 0
        self.h = h
        self.x = np.array([])

    def add_literal(self, x):
        if x > 0:
            self.x = np.append(self.x, True)
        elif x < 0:
            self.x = np.append(self.x, False)

    def remove_literal(self, i):
        self.x[i] = None