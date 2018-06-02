

def inferenceRule(rule, min, max, array_of_beliefs):
    alfa_t = 0.0
    alfa_k = 0.0
    how_many_missing_trues = 0
    how_many_missing_falses = 0
    confidence_value = rule.c
    alfa_m = (min+max)/2
    #TODO nao sei como identificar o missing
    for literals in rule.x:
        if literals:
            alfa_t += 1.0
        else:
            alfa_k += 1.0