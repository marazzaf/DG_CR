import numpy as np
import matplotlib.pyplot as plt
import sys

CG_1 = np.loadtxt('CG_CG_perf_1/perf.txt')
DG_1 = np.loadtxt('DG_CR_perf_1/perf.txt')
CG_2 = np.loadtxt('CG_CG_perf_2/perf.txt')
DG_2 = np.loadtxt('DG_CR_perf_2/perf.txt')
CG_3 = np.loadtxt('CG_CG_perf_3/perf.txt')
DG_3 = np.loadtxt('DG_CR_perf_3/perf.txt')

#nb iterations
tot_CG_1 = np.cumsum(CG_1[:,1])
tot_DG_1 = np.cumsum(DG_1[:,1])
tot_CG_2 = np.cumsum(CG_2[:,1])
tot_DG_2 = np.cumsum(DG_2[:,1])
tot_CG_3 = np.cumsum(CG_3[:,1])
tot_DG_3 = np.cumsum(DG_3[:,1])
plt.plot(CG_1[:,0]/len(CG_1[:,0]),tot_CG_1, label='CG CG 1')
plt.plot(DG_1[:,0]/len(DG_1[:,0]),tot_DG_1, label='DG CR 1')
#plt.plot(CG_2[:,0]/len(CG_2[:,0]),tot_CG_2, label='CG CG 2')
plt.plot(DG_2[:,0]/len(DG_2[:,0]),tot_DG_2, label='DG CR 2')
#plt.plot(CG_3[:,0]/len(CG_3[:,0]),tot_CG_3, label='CG CG 3')
plt.plot(DG_3[:,0]/len(DG_3[:,0]),tot_DG_3, label='DG CR 3')
plt.legend(loc='upper left')
plt.xlabel('t')
plt.ylabel('total nb iter')
plt.savefig('nb_iter.pdf')
plt.title('Total number of iterations')
plt.show()

#convergence of DG
plt.plot(DG_1[:,0]/len(DG_1[:,0]), DG_1[:,2], label='DG 1')
plt.plot(DG_2[:,0]/len(DG_2[:,0]), DG_2[:,2], label='DG 2')
plt.plot(DG_3[:,0]/len(DG_3[:,0]), DG_3[:,2], label='DG 3')
plt.legend(loc='upper left')
plt.title(r'$L^2$ error')
plt.show()
sys.exit()

plt.plot(CG[:,3], label='CG')
plt.plot(DG_CR[:,3], label='DG CR')
#plt.plot(DG_CG[:,2], label='DG CG')
plt.legend(loc='upper left')
plt.title(r'$H^1$ error')
plt.show()
