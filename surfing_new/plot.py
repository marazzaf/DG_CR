import matplotlib.pyplot as plt
import numpy as np

DG_2 = np.loadtxt('l_h_2/test_DG_4/energies.txt')
DG_5 = np.loadtxt('l_h_5/test_DG_4/energies.txt')
CG_2 = np.loadtxt('l_h_2/test_CG_44/energies.txt')
CG_5 = np.loadtxt('l_h_5/test_CG_44/energies.txt')

plt.plot(DG_2[:,0], DG_2[:,-2], '-*', label='DG $\ell/h=2$', color='#1f77b4')
plt.plot(DG_5[:,0], DG_5[:,-2], '-o', label='DG $\ell/h=5$', color='#1f77b4')
plt.plot(CG_2[:,0], CG_2[:,-2], '-*', label='CG $\ell/h=2$', color='#ff7f0e')
plt.plot(CG_5[:,0], CG_5[:,-2], '-o', label='CG $\ell/h=5$', color='#ff7f0e')
plt.xlim((0.25,0.7))
plt.legend()
plt.savefig('test.pdf')
plt.show()
