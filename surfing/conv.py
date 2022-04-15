import numpy as np
import matplotlib.pyplot as plt
import sys

c_0 = np.loadtxt('conv_0/conv.txt')
c_1 = np.loadtxt('conv_1/conv.txt')
c_2 = np.loadtxt('conv_2/conv.txt')
c_3 = np.loadtxt('conv_3/conv.txt')
c_4 = np.loadtxt('conv_4/conv.txt')

n = np.array([110605, 248030, 511597, 1057744, 2028499])
err = np.array([c_0[:,1], c_1[:,1], c_2[:,1], c_3[:,1], c_4[:,1]])
err_4 = err[:,0]
err_6 = err[:,1]
err_8 = err[:,2]

conv_4 = 2 * np.log(err_4[:-1] / err_4[1:]) / np.log(n[1:] / n[:-1])
print(conv_4)
conv_6 = 2 * np.log(err_6[:-1] / err_6[1:]) / np.log(n[1:] / n[:-1])
print(conv_6)
conv_8 = 2 * np.log(err_8[:-1] / err_8[1:]) / np.log(n[1:] / n[:-1])
print(conv_8)
sys.exit()

plt.plot(CG_1[:,1],tot_CG_1, label='CG CG 1')
plt.plot(DG_1[:,1],tot_DG_1, label='DG CR 1')
plt.legend(loc='upper left')
plt.xlabel('t')
plt.ylabel('total nb iter')
plt.title('Total number of iterations - 1')
plt.savefig('iterations_1.pdf')
plt.show()
