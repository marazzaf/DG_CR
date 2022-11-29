import matplotlib.pyplot as plt
import numpy as np

CG_2 = np.loadtxt('test_CG_2/energies.txt')
DG_2 = np.loadtxt('test_DG_2/energies.txt')
CG_3 = np.loadtxt('test_CG_3/energies.txt')
DG_3 = np.loadtxt('test_DG_3/energies.txt')

plt.plot(CG_2[:,0], CG_2[:,-2], '-*', label='CG 2')
plt.plot(DG_2[:,0], DG_2[:,-2], '-*', label='DG 2')
plt.plot(CG_3[:,0], CG_3[:,-2], '-*', label='CG 3')
plt.plot(DG_3[:,0], DG_3[:,-2], '-*', label='DG 3')
plt.legend()
plt.show()
