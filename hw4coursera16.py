import numpy as np

# read files
f = open('hw4_train.dat.txt', 'r')
trainX = []
trainY = []
for s in f:
    nx = [1]
    tmp = list(map(float, s.split()))
    for i in range(len(tmp)-1):
        nx.append(tmp[i])
    trainX.append(nx)
    trainY.append(tmp[-1])

f = open('hw4_test.dat.txt', 'r')
testX = []
testY = []
for s in f:
    nx = [1]
    tmp = list(map(float, s.split()))
    for i in range(len(tmp)-1):
        nx.append(tmp[i])
    testX.append(nx)
    testY.append(tmp[-1])

validX = trainX[120:]
validY = trainY[120:]
trainX = trainX[:120]
trainY = trainY[:120]

validX = np.array(validX)
validY = np.array(validY)
trainX = np.array(trainX)
trainY = np.array(trainY)
testX = np.array(testX)
testY = np.array(testY)

# help functions
def solve(X, Y, r):
    w = np.matmul(np.transpose(X), X) + r*np.identity(3)
    w = np.linalg.pinv(w)
    w = np.matmul(w, np.transpose(X))
    w = np.matmul(w, Y)
    return w
     
def test(w, X, Y):
    y = np.sign(np.matmul(w, np.transpose(X)))
    return np.sum(np.transpose(y) != Y) / len(Y)

# solve
etrain = np.Inf
eva = np.Inf
eout = np.Inf
mn = 0
for reg in range(-10, 3):
    Wreg = solve(trainX, trainY, 10**reg)
    err = test(Wreg, trainX, trainY)
    if err <= etrain:
        etrain = err
        eva = test(Wreg, validX, validY)
        eout = test(Wreg, testX, testY)
        mn = reg

print(mn, etrain, eva, eout)

