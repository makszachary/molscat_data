import numpy as np
from pathlib import Path

from matplotlib import pyplot as plt

from _molscat_data.physical_constants import hartree_in_inv_cm, fine_structure_constant

so_coupling_path = Path(__file__).parents[1] / 'data' / 'so_coupling' / 'lambda_SO_a_SrRb+_MT_original.dat'
so_coupling = np.loadtxt(so_coupling_path, dtype = float)

so_coupling[:,1] /= hartree_in_inv_cm
so_coupling[:,1] += -fine_structure_constant**2 / so_coupling[:,0]**3

plt.scatter(so_coupling[:,0], so_coupling[:,1])
plt.show()



# new_so_coupling_path = so_coupling_path.with_stem('lambda_SO+SS_MT_Hartrees')
# np.savetxt(new_so_coupling_path, so_coupling, fmt = ['%.2f','%.15e'])