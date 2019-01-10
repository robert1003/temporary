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
    y = np.sign(np.matmul(X, np.transpose(w)))
    return np.sum(y != Y) / len(Y)

# parameters
reg = 10

# solve
Wreg = solve(trainX, trainY, reg)

print('Ein', test(Wreg, trainX, trainY))
print('Eout', test(Wreg, testX, testY))
