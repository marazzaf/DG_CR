import numpy as np
import matplotlib.pyplot as plt

CG = np.loadtxt('CG_CG_1/perf.txt')
DG_CR = np.loadtxt('DG_CR_1/perf.txt')
DG_CG = np.loadtxt('DG_CG_1/perf.txt')

tot_CG = np.cumsum(CG[:,0])
tot_DG_CR = np.cumsum(DG_CR[:,0])
tot_DG_CG = np.cumsum(DG_CG[:,0])

plt.plot(tot_CG, label='CG')
plt.plot(tot_DG_CR, label='DG CR')
plt.plot(tot_DG_CG, label='DG CG')
plt.legend(loc='upper left')
plt.xlabel('t')
plt.ylabel('total nb iter')
plt.savefig('nb_iter.pdf')=
plt.title('Total number of iterations')
plt.show()

plt.plot(CG[:,1], label='CG')
plt.plot(DG_CR[:,1], label='DG CR')
plt.plot(DG_CG[:,1], label='DG CG')
plt.legend(loc='upper left')
plt.title(r'$L^2$ error')
plt.show()

plt.plot(CG[:,2], label='CG')
plt.plot(DG_CR[:,2], label='DG CR')
plt.plot(DG_CG[:,2], label='DG CG')
plt.legend(loc='upper left')
plt.title(r'$H^1$ error')
plt.show()
