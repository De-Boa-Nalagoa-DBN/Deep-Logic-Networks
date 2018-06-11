from sklearn import preprocessing
from rule import Rule
import numpy as np

def inf1(rule, input):
    positives = np.sum(input[rule.x == True])
    negatives = np.sum(input[rule.x == False])

    return (positives - negatives) * rule.c

def quantitativeInference(ruleSet, input_data):
    current_data = input_data
    for _, rules in enumerate(ruleSet):
        alfas = [inf1(r, current_data) for r in rules]
        alfas = np.asarray(alfas)
        # print(alfas.shape)
        #alfas = alfas / np.max(alfas)
        alfas = alfas / np.linalg.norm(alfas, ord=np.inf, axis=0, keepdims=True)
        current_data = alfas

    return current_data

def main():
    r = Rule(0)
    r.c = 1.5
    r.x = np.array([True, False, True])
    input = np.array([1, 0.5, 1])
    output = quantitativeInference([[r]], input)
    print(output)


if __name__ == '__main__':
    main()