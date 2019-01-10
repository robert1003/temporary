import numpy as np

# read files
f = open('hw3_train.dat.txt', 'r')
trainX = []
trainY = []
for s in f:
    nx = [1]
    tmp = list(map(float, s.split()))
    for i in range(len(tmp)-1):
        nx.append(tmp[i])
    trainX.append(nx)
    trainY.append(tmp[-1])

f = open('hw3_test.dat.txt', 'r')
testX = []
testY = []
for s in f:
    nx = [1]
    tmp = list(map(float, s.split()))
    for i in range(len(tmp)-1):
        nx.append(tmp[i])
    testX.append(nx)
    testY.append(tmp[-1])

trainX = np.array(trainX)
trainY = np.array(trainY)
testX = np.array(testX)
testY = np.array(testY)

# some functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def grad(w, X, Y):
    tmp = 0.0
    for x, y in zip(X, Y):
        tmp += sigmoid(-y*np.dot(w, x))*(-y*x)
    tmp /= len(X)
    return tmp

def accuracy(w, X, Y):
    err = 0
    for x, y in zip(X, Y):
        err += int(np.sign(np.dot(w, x))) != int(y)
    return err / len(X)

# train
epoch = 2000
eta1 = 0.001
eta2 = 0.01
w1 = np.zeros(21)
w2 = np.zeros(21)
ein1 = []
ein2 = []
eout1 = []
eout2 = []


i = 0
for e in range(epoch):
    w1 = w1 - eta1*grad(w1, trainX[i:i+1], trainY[i:i+1])
    w2 = w2 - eta2*grad(w2, trainX, trainY)
    i = (i + 1) % len(trainX)
    ein1.append(accuracy(w1, trainX, trainY))
    ein2.append(accuracy(w2, trainX, trainY))
    eout1.append(accuracy(w1, testX, testY))
    eout2.append(accuracy(w2, testX, testY))

import matplotlib.pyplot as plt
t = range(1, epoch+1)
plt.plot(t, ein1, label='SGD')
plt.plot(t, ein2, label='GD')
plt.title('Ein')
plt.legend()
plt.show()

plt.plot(t, eout1, label='SGD')
plt.plot(t, eout2, label='GD')
plt.title('Eout')
plt.legend()
plt.show()

