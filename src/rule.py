import numpy as np

class Rule:
    def __init__(self, h):
        self.c = 0
        self.h = h
        self.x = []

    def add_literal(self, x):
        if x > 0:
            self.x.append(True)
        elif x < 0:
            self.x.append(False)

    def remove_literal(self, i):
        self.x[i] = None