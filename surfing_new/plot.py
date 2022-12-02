import matplotlib.pyplot as plt
import numpy as np

CG_4 = np.loadtxt('test_CG_44/energies.txt')
DG_4 = np.loadtxt('test_DG_4/energies.txt')
CG_3 = np.loadtxt('test_CG_33/energies.txt')
DG_3 = np.loadtxt('test_DG_3/energies.txt')

plt.plot(CG_4[:,0], CG_4[:,-2], '-*', label='CG 4')
plt.plot(DG_4[:,0], DG_4[:,-2], '-*', label='DG 4')
plt.plot(CG_3[:,0], CG_3[:,-2], '-*', label='CG 3')
plt.plot(DG_3[:,0], DG_3[:,-2], '-*', label='DG 3')
plt.legend()
plt.show()
