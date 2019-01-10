import random as rand

dataset = []
testset = []
wpocket = [0, 0, 0, 0, 0]
er = 100000

# get dataset
while True:
	try:
		s = list(map(float, input().split()))
		dataset.append([(1, s[0], s[1], s[2], s[3]), int(s[4])])
	except:
		break

for x, s in dataset:
	print(x, s)

# get testset
while True:
	try:
		s = list(map(float, input().split()))
		testset.append([(1, s[0], s[1], s[2], s[3]), int(s[4])])
	except:
		break

for x, s in testset:
	print(x, s)

# dot
def dot(a, b):
	val = 0
	for i in range(5):
		val += a[i]*b[i]

	if val <= 0:
		return -1
	else:
		return 1

# update wpocket
def update(w):
	global er
	global wpocket
	global dataset

	erc = 0
	for x, s in dataset:
		if int(dot(w, x)) != s:
			erc += 1

	#print(erc, er)
	if erc < er:
		er = erc
		#wpocket = w
		for i in range(5):
			wpocket[i] = w[i]
		#print(wpocket, er)
		return 1
	else:
		return 0

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
	global wpocket
	global er

	w = [0, 0, 0, 0, 0]
	wpocket = [0, 0, 0, 0, 0]
	er = 100000
	cnt = 0

	while cnt < 100:
		#print(cnt)
		now = check_error(w)
		if now == None:
			break
		else:
			x, s = now
			for i in range(5):
				w[i] += s*x[i]

			update(w)
			cnt += 1
			#cnt += int(update(w))

# test
def test():
	global wpocket
	global testset

	erc = 0

	for x, s in testset:
		if int(dot(wpocket, x)) != s:
			erc += 1

	return erc/len(testset)


ans = 0.0
for i in range(2000):
	print(i)
	rand.seed(i)
	seq.sort()
	rand.shuffle(seq)

	pla()
	ans += test()
print(ans/2000)






