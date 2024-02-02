from pathlib import Path
import numpy as np

from _molscat_data.effective_probability import p0, effective_probability

pmf_path = Path(__file__).parents[1] / 'data' / 'pmf' / 'N_pdf_logic_params_EMM_500uK.txt'
pmf_array = np.loadtxt(pmf_path)

names = ['single_ion_hpf.dat', 'single_ion_cold_higher.dat', 'single_ion_cold_lower.dat']
for name in names:
    peff_path = Path(__file__).parents[1] / 'data' / 'exp_data' / name
    p0_path = peff_path.parent / ('p0_' + peff_path.name)

    peff_exp, peff_std = np.loadtxt(peff_path)
    print(peff_exp)
    print(peff_std)
    dpeff = 1e-3
    p0_exp = p0(peff_exp, pmf_array=pmf_array)
    p0_std = (p0(peff_exp+dpeff/2, pmf_array=pmf_array, ignore_negative = True)-p0(peff_exp-dpeff/2, pmf_array=pmf_array, ignore_negative = True))/dpeff * peff_std
    print(p0_exp,'\n',p0_std)
    p0_data = np.around([p0_exp, p0_std], decimals = 4)
    print(p0_data)

    np.savetxt(p0_path, p0_data, fmt = '%.4f', header = f"The short-range probabilities and their standard deviations calculated from {name} using probability mass function stored in {pmf_path}.")


# yy = np.array([[1,2,4,2],
#               [3,4,1,1]])
# xx = np.array([[1, 4, 7, 10],
#               [2, 5, 8, 11]])

# fig, ax = plt.subplots()
# for x, y in zip(xx, yy):
#     ax.bar(x, y, width = 1)#, hatch = theory_hatch, **bars_formatting)
# plt.show()