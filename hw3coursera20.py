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
eta = 0.001
w = np.zeros(21)

i = 0
for e in range(epoch):
    w = w - eta*grad(w, trainX[i:i+1], trainY[i:i+1])
    i = (i + 1) % len(trainX)

# test
print(accuracy(w, testX, testY))
