import random as rand
dataset = []
while True:
    try:
        s = list(map(float, input().split()))
        s[4] = int(s[4])
        dataset.append([(1, s[0], s[1], s[2], s[3]), s[4]])
    except:
        break

for x, s in dataset:
    print(x, s, sep=": ")

def dot(a, b):
    val = 0
    for i in range(5):
        val += a[i]*b[i]
    
    if val <= 0:
        return -1
    else:
        return 1

seq = [i for i in range(len(dataset))]
now = -1
wpocket = [0, 0, 0, 0, 0]
er = 100000000

def better(w):
    erc = 0
    global er
    global wpocket
    for i in range(len(dataset)):
        x, s = dataset[i]
        if int(dot(w, x)) != s:
            erc += 1
    if erc < er:
        wpocket = w
        er = erc
        return 1
    else:
        return 0

def check_error(w):
    result = None
    for i in range(len(dataset)):
        global now
        now = (now+1)%len(dataset)
        x, s = dataset[seq[now]]
        if int(dot(w, x)) != s:
            result = x, s
            return result
    return None

def pla():
    w = [0, 0, 0, 0, 0]
    cnt = 0
    nnt = 0
    while nnt < 100:
#        print(w)
#        print(wpocket, er)
        x, s = check_error(w)
        for i in range(5):
            w[i] += s*x[i]
        cnt += better(w)
        nnt += 1
    return w

testset = []
while True:
    try:
        s = list(map(float, input().split()))
        s[4] = int(s[4])
        testset.append([(1, s[0], s[1], s[2], s[3]), s[4]])
    except:
        break
for x, s in testset:
    print(x, s, sep=": ")


def test(w):
    eror = 0
    for x, s in testset:
        if int(dot(w, x)) != s:
            eror += 1
    return eror/len(testset)

ave = 0.0
for t in range(2000):
    print("start", t)
    rand.seed(t)
    rand.shuffle(seq)
    wpocket = [0, 0, 0, 0, 0]
    er = 100000000
    now = -1
    pla()
    ave += test(wpocket)

print(ave/2000)

