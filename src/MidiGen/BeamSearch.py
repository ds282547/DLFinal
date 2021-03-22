

import queue
import math
import numpy as np

from operator import itemgetter

'''
p = [2.4270e-05, 2.6262e-04, 1.0343e-05, 9.2451e-06, 3.2495e-06, 7.5959e-05,
        1.2512e-04, 7.5113e-06, 5.9078e-06, 2.6710e-05, 1.6250e-06, 5.8081e-05,
        1.4642e-04, 2.8621e-01, 1.5349e-05, 2.9773e-06, 4.8595e-05, 9.3127e-05,
        2.4007e-04, 6.2231e-05, 1.1806e-04, 5.2540e-06, 7.2460e-05, 4.0042e-04,
        7.1756e-04, 9.0123e-01, 3.9191e-04, 8.2055e-05, 6.3167e-05, 3.3506e-04,
        3.2966e-04, 7.4034e-03, 8.2141e-02, 6.5283e-05, 5.7975e-06, 9.6613e-04,
        2.1889e-04, 4.3158e-01, 6.3176e-05, 4.1968e-04, 3.3577e-03, 9.7032e-01,
        3.0159e-03, 3.6703e-04, 2.0678e-01, 4.1462e-03, 9.1578e-04, 7.5797e-01,
        6.4497e-04, 1.6685e-01, 2.5449e-03, 6.1654e-03, 1.5348e-03, 3.4219e-02,
        1.7323e-03, 1.6405e-04, 5.8065e-03, 3.7464e-04, 1.0240e-03, 5.1637e-02,
        2.4140e-04, 9.9571e-03, 8.4737e-05, 2.3820e-03, 9.5031e-03, 5.6963e-03,
        2.8475e-03, 2.7521e-04, 4.9091e-03, 3.1891e-03, 9.3843e-04, 7.5242e-03,
        2.7736e-04, 7.6554e-03, 6.8158e-05, 9.2036e-04, 9.2625e-04, 1.0464e-03,
        8.0872e-04, 2.0244e-04, 1.3363e-03, 2.1324e-04, 3.0645e-04, 6.7789e-04,
        4.5676e-05, 1.7986e-05, 3.9160e-06, 9.8543e-07 ]

p2 = [7.3006e-06, 4.6595e-04, 8.1498e-06, 2.6975e-05, 5.5884e-06, 1.2436e-04,
        8.2784e-05, 1.0573e-05, 1.4175e-05, 6.7220e-06, 4.8178e-06, 1.8868e-05,
        4.5990e-05, 1.2736e-01, 1.5585e-05, 1.0202e-04, 2.6020e-04, 1.3378e-04,
        4.9625e-04, 2.9341e-04, 6.3948e-05, 9.5394e-06, 4.6708e-04, 2.1675e-04,
        3.8900e-04, 8.9714e-01, 5.3071e-04, 3.1290e-04, 2.0073e-04, 5.9473e-04,
        9.0085e-05, 1.2341e-03, 7.3088e-02, 8.3151e-05, 3.5402e-03, 2.7473e-02,
        1.8874e-04, 5.1122e-01, 2.4321e-04, 1.7292e-04, 1.4838e-03, 9.8718e-01,
        3.3366e-04, 4.1019e-03, 9.5742e-01, 6.8229e-04, 2.6781e-03, 6.1645e-01,
        6.2873e-04, 3.5531e-01, 4.4022e-03, 1.9728e-03, 4.1170e-03, 1.6854e-01,
        3.0806e-04, 8.7755e-04, 1.9463e-02, 3.9721e-04, 1.5932e-03, 7.4622e-02,
        1.9593e-03, 5.6597e-03, 2.7996e-04, 2.9089e-03, 1.2705e-03, 6.8507e-03,
        2.4292e-04, 3.4246e-04, 4.3668e-03, 7.9446e-05, 1.5993e-04, 8.9403e-04,
        3.8961e-04, 1.2121e-03, 1.0097e-04, 3.5136e-04, 2.4292e-04, 1.0826e-03,
        1.6098e-04, 1.2090e-04, 1.0181e-03, 5.1341e-05, 9.3211e-05, 9.3768e-05,
        1.3803e-05, 7.7959e-06, 3.7040e-06, 3.3357e-06]
'''

def find_beam(p, beamNum):
    # p = np.array(p)

    v0 = (p > 0.3).astype(bool)

    l0 = 0
    L = []
    for i in range(88):
        # l0 += math.log(max(p[i], 1-p[i]))
        if p[i] == 0:
            L.append(float("-inf"))
        elif p[i] == 1:
            L.append(1.0)
        else:
            L.append(math.fabs(math.log(p[i]/(1-p[i]))))

    L = np.array(L)
    R, L = zip(*sorted(enumerate(L), key=itemgetter(1)))

    R = list(R)
    L = list(L)

    #print(l0)
    # print('V 0')
    v0_set = []
    for i in range(88):
        if v0[i]:
            v0_set.append(i)

    #print(L)

    q = queue.PriorityQueue()


    v = [1] + [0] * 87

    q.put((L[0], v.copy()))
    del v

    N = 88
    # print(l0)
    selected = [v0_set]
    while not q.empty():

        l, v = q.get()

        vb = np.array(v, dtype=bool)
        vRb = np.zeros(vb.size)

        for i in range(88):
            if vb[i]:
                vRb[R[i]] = True

        X = np.logical_xor(vRb, v0)
        check = []
        for i in range(88):
            if X[i]:
                check.append(i)
                #print(i, end=' ')
        #print('')


        #print(l0 - l)


        for i in range(87, -1, -1):
            if v[i] == 1:
                break

        if i < N - 2:
            v[i+1] = 1
            q.put((l + L[i+1], v.copy()))
            v[i] = 0
            q.put((l + L[i+1] - L[i], v.copy()))
            del v
        # print(q)
        selected.append(check)
        if len(selected) >= beamNum:
            return  selected

def find_beam2(p, thres, beamNum, int_thres=None, z=0.3,b=0.0):
    # p = np.array(p)

    # [0,1] map to [1-z, z]
    thres = (1 - thres) * (1 - z*2) + z + b

    if not int_thres is None:
        thres = int_thres

    v0 = (p > thres).astype(bool)

    l0 = 0
    L = []
    for i in range(88):
        if p[i] > 0:
            L.append(math.fabs(math.log(p[i]) - math.log(1 - p[i])))
        else:

            L.append(float("-inf"))

    L = np.array(L)
    R, L = zip(*sorted(enumerate(L), key=itemgetter(1)))

    R = list(R)
    L = list(L)

    #print(l0)
    # print('V 0')
    v0_set = []
    for i in range(88):
        if v0[i]:
            v0_set.append(i)

    #print(L)

    q = queue.PriorityQueue()


    v = [1] + [0] * 87

    q.put((L[0], v.copy()))
    del v

    N = 88
    # print(l0)
    selected = [v0_set]
    while not q.empty():

        l, v = q.get()

        vb = np.array(v, dtype=bool)
        vRb = np.zeros(vb.size)

        for i in range(88):
            if vb[i]:
                vRb[R[i]] = True

        X = np.logical_xor(vRb, v0)
        check = []
        for i in range(88):
            if X[i]:
                check.append(i)
                #print(i, end=' ')
        #print('')

        #print(l0 - l)

        for i in range(87, -1, -1):
            if v[i] == 1:
                break

        if i < N - 2:
            v[i+1] = 1
            q.put((l + L[i+1], v.copy()))
            v[i] = 0
            q.put((l + L[i+1] - L[i], v.copy()))
            del v
        # print(q)
        selected.append(check)
        if len(selected) >= beamNum:
            return  selected


def cal_prob(output, beam):
    output = 1 - output

    for b in beam:
        output[b] = 1 - output[b]

    return output