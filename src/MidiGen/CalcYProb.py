import numpy as np
import math as m


x = np.load('CalcProb_TotalN_19380449.npy')
N = 19380449

PY = np.zeros((88, 2))
for i in range(x.shape[0]):
    #p = x[i] / N

    logp = m.log(x[i]) - m.log(N)

    log1mp = m.log(N - x[i]) - m.log(N)
    PY[i][0] = log1mp
    PY[i][1] = logp


def calLogProb(beam):
    choose = np.zeros(88, dtype='int')

    choose[beam] = 1
    sum = 0
    for i in range(88):
        sum += PY[i, choose[i]]
    #print(sum)
    return sum

