import numpy as np
import matplotlib.pyplot as plt

ld = np.loadtxt('cluster/ld.txt')

plt.plot(ld[:,0], abs(ld[:,1]), '-')
plt.xlim(0,0.12)
plt.ylim(0,0.05)
plt.xlabel('Displacement [mm]')
plt.ylabel('Force [kN]')
plt.grid(True, linestyle='--')
plt.savefig('ld_beam.pdf')
plt.show()
