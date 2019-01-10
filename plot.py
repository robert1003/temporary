import random as rand

dataset = []

# get dataset
while True:
	try:
		s = list(map(float, input().split()))
		dataset.append([(1, s[0], s[1], s[2], s[3]), int(s[4])])
	except:
		break

# dot
def dot(a, b):
	val = 0
	for i in range(5):
		val += a[i]*b[i]

	if val <= 0:
		return -1
	else:
		return 1

seq = [i for i in range(len(dataset))]
idx = -1

# find error
def check_error(w):
	global idx
	global seq
	global dataset

	for i in range(len(dataset)):
		idx = (idx+1)%len(dataset)
		x, s = dataset[seq[idx]]
		if int(dot(w, x)) != s:
			return (x, s)

	return None

# PLA
def pla():
	w = [0, 0, 0, 0, 0]
	cnt = 0

	while True:
		#print(cnt)
		now = check_error(w)
		if now == None:
			break
		else:
			cnt += 1
			x, s = now
			for i in range(5):
				w[i] += s*x[i]

	return cnt

# main
ans = []
cnt = 0

for i in range(1126):
	rand.seed(i)
	seq.sort()
	rand.shuffle(seq)

	ans.append(pla())
	cnt += ans[-1]
cnt /= 1126

# draw
import matplotlib.pyplot as plt

x = [i for i in range(100)]
y = [0 for i in range(100)]
for i in ans:
	y[i] += 1

plt.bar(x, y)
plt.xlabel('number of updates')
plt.ylabel('frequency of number')
name = 'average number of updates before halts: ' + str(cnt)
plt.title(name)
plt.show()





