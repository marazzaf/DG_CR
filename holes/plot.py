import numpy as np
import matplotlib.pyplot as plt

ld = np.loadtxt('cluster/ld.txt')
x = np.arange(0, 0.201, 1e-2)
plt.plot(x, 0.44125/0.2*x, '-', color='blue')
plt.plot(ld[:,0], abs(ld[:,1]), '-', color='blue')
plt.xlim(0,1.25)
plt.ylim(0,0.7)
plt.xlabel('Displacement [mm]')
plt.ylabel('Force [kN]')
plt.grid(True, linestyle='--')
plt.savefig('ld_holes.pdf')
plt.show()
