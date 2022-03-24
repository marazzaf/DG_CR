import numpy as np
import matplotlib.pyplot as plt

CG = np.loadtxt('CG_CG_1/perf.txt')
DG = np.loadtxt('DG_CR_1/perf.txt')

tot_CG = np.cumsum(CG[:,0])
tot_DG = np.cumsum(DG[:,0])

plt.plot(tot_CG, label='CG')
plt.plot(tot_DG, label='DG')
plt.legend(loc='upper left')
plt.xlabel('t')
plt.ylabel('total nb iter')
plt.savefig('nb_iter.pdf')
plt.show()
