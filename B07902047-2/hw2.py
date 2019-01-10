import numpy as np
import random as rnd
import matplotlib.pyplot as plt

EIN = []
EOUT = []
for cnt in range(1000):
    rnd.seed(cnt)
    #generate dataset
    x = np.random.rand(1, 20)*2-1
    y = np.sign(x)
    for i in range(len(y[0])):
        p = rnd.random()
        if p < 0.2:
            y[0][i] = -y[0][i]
    #decision stump algorithm
    h = np.sort(x)
    h = np.append(h, 1)
    h = np.append(h, -1)
    ein = 1e9
    s = 0
    th = 0
    for i in range(len(h)-1):
        theta = (h[i-1]+h[i])/2
        err = 0
        for j in range(len(x[0])):
            hx = int(np.sign(x[0][j]-theta));
            if hx != (int)(y[0][j]):
                err += 1
        if err < ein:
            ein = err
            s = 1
            th = theta
    for i in range(len(h)-1):
        theta = (h[i-1]+h[i])/2
        err = 0
        for j in range(len(x[0])):
            hx = -int(np.sign(x[0][j]-theta));
            if hx != (int)(y[0][j]):
                err += 1
        if err < ein:
            ein = err
            s = -1
            th = theta
    EIN.append(ein/20)
    EOUT.append(0.5+0.3*s*(abs(th)-1))

X = [i-j for (i, j) in zip(EIN, EOUT)]
plt.hist(X)
plt.show()