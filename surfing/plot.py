import numpy as np
import matplotlib.pyplot as plt
import sys

CG_1 = np.loadtxt('CG_CG_perf_1/perf.txt')
DG_1 = np.loadtxt('DG_CR_perf_1/perf.txt')
CG_2 = np.loadtxt('CG_CG_perf_2/perf.txt')
DG_2 = np.loadtxt('DG_CR_perf_2/perf.txt')
CG_3 = np.loadtxt('CG_CG_perf_3/perf.txt')
DG_3 = np.loadtxt('DG_CR_perf_3/perf.txt')
CG_4 = np.loadtxt('CG_CG_perf_4/perf.txt')
DG_4 = np.loadtxt('DG_CR_perf_4/perf.txt')

#nb iterations
tot_CG_1 = np.cumsum(CG_1[:,2])
tot_DG_1 = np.cumsum(DG_1[:,2])
tot_CG_2 = np.cumsum(CG_2[:,2])
tot_DG_2 = np.cumsum(DG_2[:,2])
tot_CG_3 = np.cumsum(CG_3[:,2])
tot_DG_3 = np.cumsum(DG_3[:,2])
tot_CG_4 = np.cumsum(CG_4[:,2])
tot_DG_4 = np.cumsum(DG_4[:,2])
plt.plot(CG_1[:,1],tot_CG_1, label='CG CG 1')
plt.plot(DG_1[:,1],tot_DG_1, label='DG CR 1')
plt.legend(loc='upper left')
plt.xlabel('t')
plt.ylabel('total nb iter')
plt.title('Total number of iterations - 1')
plt.savefig('iterations_1.pdf')
plt.show()

plt.plot(CG_2[:,1],tot_CG_2, label='CG CG 2')
plt.plot(DG_2[:,1],tot_DG_2, label='DG CR 2')
plt.legend(loc='upper left')
plt.xlabel('t')
plt.ylabel('total nb iter')
plt.title('Total number of iterations - 2')
plt.show()

plt.plot(CG_3[:,1],tot_CG_3, label='CG CG 3')
plt.plot(DG_3[:,1],tot_DG_3, label='DG CR 3')
plt.legend(loc='upper left')
plt.xlabel('t')
plt.ylabel('total nb iter')
plt.title('Total number of iterations - 3')
plt.show()

plt.plot(CG_4[:,1],tot_CG_4, label='CG CG 4')
plt.plot(DG_4[:,1],tot_DG_4, label='DG CR 4')
plt.legend(loc='upper left')
plt.xlabel('t')
plt.ylabel('total nb iter')
plt.title('Total number of iterations - 4')
plt.show()

#convergence of DG
plt.plot(DG_1[:,1], DG_1[:,3], label='DG 1')
plt.plot(DG_2[:,1], DG_2[:,3], label='DG 2')
plt.plot(DG_3[:,1], DG_3[:,3], label='DG 3')
plt.plot(DG_4[:,1], DG_4[:,3], label='DG 4')
plt.legend(loc='upper left')
plt.title(r'$L^2$ error')
plt.show()
plt.savefig('error_l2.pdf')

plt.plot(DG_1[:,1], DG_1[:,3], label='DG 1')
plt.plot(DG_2[:,1], DG_2[:,3], label='DG 2')
plt.plot(DG_3[:,1], DG_3[:,3], label='DG 3')
plt.plot(DG_4[:,1], DG_4[:,3], label='DG 4')
plt.plot(CG_1[:,1], CG_1[:,3], label='CG 1')
plt.plot(CG_2[:,1], CG_2[:,3], label='CG 2')
plt.plot(CG_3[:,1], CG_3[:,3], label='CG 3')
plt.plot(CG_4[:,1], CG_4[:,3], label='CG 4')
plt.legend(loc='upper left')
plt.title(r'$L^2$ error')
plt.show()

plt.plot(DG_1[:,1], DG_1[:,4], label='DG 1')
plt.plot(DG_2[:,1], DG_2[:,4], label='DG 2')
plt.plot(DG_3[:,1], DG_3[:,4], label='DG 3')
plt.plot(DG_4[:,1], DG_4[:,4], label='DG 4')
plt.legend(loc='upper left')
plt.title(r'$H^1$ error')
plt.show()
