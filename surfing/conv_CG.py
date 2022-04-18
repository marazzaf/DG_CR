import numpy as np

data = np.loadtxt('conv_CG.txt')

n = data[:,0]
err = data[:,1]

conv = 2 * np.log(err[:-1] / err[1:]) / np.log(n[1:] / n[:-1])
print(conv)
