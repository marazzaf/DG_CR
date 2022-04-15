import numpy as np
import matplotlib.pyplot as plt
import sys

c_0 = np.loadtxt('conv_0/conv.txt')
c_1 = np.loadtxt('conv_1/conv.txt')
c_2 = np.loadtxt('conv_2/conv.txt')
c_3 = np.loadtxt('conv_3/conv.txt')
c_4 = np.loadtxt('conv_4/conv.txt')

n_0 = 110605
n_1 = 248030
n_2 = 511597
n_3 = 1057744
n_4 = 2028499

sys.exit()

plt.plot(CG_1[:,1],tot_CG_1, label='CG CG 1')
plt.plot(DG_1[:,1],tot_DG_1, label='DG CR 1')
plt.legend(loc='upper left')
plt.xlabel('t')
plt.ylabel('total nb iter')
plt.title('Total number of iterations - 1')
plt.savefig('iterations_1.pdf')
plt.show()
