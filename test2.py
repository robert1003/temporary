import random as rand
dataset = []
while True:
    try:
        s = list(map(float, input().split()))
        s[4] = int(s[4])
        dataset.append([(1, s[0], s[1], s[2], s[3]), s[4]])
    except:
        break

#for x, s in dataset:
#    print(x, s, sep=": ")

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
    while check_error(w) is not None:
       # print(w)
        x, s = check_error(w)
        cnt += 1
        for i in range(5):
            w[i] += 0.5*s*x[i]
 #   print(w)
 #   print(cnt)
    return cnt

cnt = 0
for t in range(2000):
    now = -1
    print(t)
    rand.seed(t)
    rand.shuffle(seq)
    cnt += pla()

print(cnt/2000)
