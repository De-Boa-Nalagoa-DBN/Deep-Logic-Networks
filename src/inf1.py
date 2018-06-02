def inferenceRule(hipothese, missing_rule, min, max):
    sum_alfa_t = 0.0
    sum_alfa_k = 0.0
    alfa_m = (min+max)/2
    confidenceValue = hipothese.c
    for literal in hipothese.x:
        if literal:
            sum_alfa_t += 1
        else:
            sum_alfa_k += 1
    alfa_h = confidenceValue*(sum_alfa_t-sum_alfa_k)
    # TODO como lidar com missing values?
    # é preciso saber quantos missing values TRUES e FALSES há
    # na hipotese. Porem a hipotese contem valores
    # trues e falses(em x), mas nao contem quais deles
    # estao missing.