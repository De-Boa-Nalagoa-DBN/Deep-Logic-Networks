import numpy as np

#ACHO QUE VAMOS PRECISAR DESSA CLASSE
class Belief:
    def __init__(self):
        self.alfa = 0
        self.x = None   #can be true or false

    def setConfidenceValue(self, newConfidence):
        self.c = newConfidence

    def setLiteral(self, literal):
        self.x = literal

    def getConfidecneValue(self):
        return self.c

    def getLiteral(self):
        return self.x

