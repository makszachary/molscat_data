import subprocess
import os
import re

from pathlib import Path

import numpy as np

from matplotlib import pyplot as plt
from _molscat_data.effective_probability import effective_probability, p0

singlet_phase, triplet_phase = 0.04, 0.24

def plot_and_save_p0_Cso(singlet_phase, triplet_phase):
    so_scaling_values = (1e-4, 1e-3, 1e-2, 0.25, 0.5, 0.75, 1.0)

    array_paths = ( Path(__file__).parents[1] / 'data_produced' / 'arrays' / 'RbSr+_tcpld_so_scaling' / '100_E' / f'{singlet_phase:.4f}_{triplet_phase:.4f}' / 'dLMax_4' / f'out_state_res_{so_scaling:.4f}_hpf.txt' for so_scaling in so_scaling_values )
    output_state_resolved_arrays = list( np.loadtxt(array_path).reshape(3,2,5) for array_path in array_paths )
    # image_path = Path(__file__).parents[1] / 'plots' / 'probability_scaling' / 'so_scaling' / f'{singlet_phase:.4f}_{triplet_phase:.4f}.png'
    image_path = Path(__file__).parents[1] / 'plots' / 'probability_scaling' / 'so_scaling' / f'p0_{singlet_phase:.4f}_{triplet_phase:.4f}.png'
    image_path.parent.mkdir(parents = True, exist_ok = True)

    ss_dominated_rates = np.fromiter( (array.sum(axis = (0,1))[4] for array in output_state_resolved_arrays), dtype = float )
    ss_dominated_rates_ratio = ss_dominated_rates / max(ss_dominated_rates)
    xx = np.logspace(-4, 1, 100)

    print(ss_dominated_rates)
    fig, ax = plt.subplots()

    # ax.scatter(so_scaling_values, ss_dominated_rates_ratio, s = 4**2, color = 'k', marker = 'o', label = '$p_0(c_\mathrm{so}) / p_0(c_\mathrm{so} = 1)$')
    # ax.plot(xx, xx**4, color = 'red', linewidth = 1, linestyle = '--', label = '$c_\mathrm{so}^4$')
    # ax.plot(xx, xx**3, color = 'orange', linewidth = 1, linestyle = '--', label = '$c_\mathrm{so}^3$')

    ax.scatter(so_scaling_values, ss_dominated_rates, s = 4**2, color = 'k', marker = 'o', label = '$p_0(c_\mathrm{so}) / p_0(c_\mathrm{so} = 1)$')
    ax.plot(xx, max(ss_dominated_rates) * xx**4, color = 'red', linewidth = 1, linestyle = '--', label = '$c_\mathrm{so}^4$')
    ax.plot(xx, max(ss_dominated_rates) * xx**3, color = 'orange', linewidth = 1, linestyle = '--', label = '$c_\mathrm{so}^3$')

    ax.set_yscale('log')
    ax.set_xscale('log')
    # ax.set_ylim(0.1*min(ss_dominated_rates_ratio), 10*max(ss_dominated_rates_ratio))
    ax.set_ylim(0.1*min(ss_dominated_rates), 10*max(ss_dominated_rates))
    ax.set_xlim(0.1*min(so_scaling_values), 10*max(so_scaling_values))
    ax.set_xlabel('spin-orbit coupling scaling factor $c_\mathrm{so}$')
    # ax.set_title('Probability of the hyperfine energy release for $\left|2,2\\right>\hspace{0.2}\left|\\hspace{-.2}\\uparrow\\hspace{-.2}\\right>$ initial state\n(relative to the result for the original spin-orbit coupling from M.T.)', fontsize = 10)
    ax.set_title('Probability of the hyperfine energy release for $\left|2,2\\right>\hspace{0.2}\left|\\hspace{-.2}\\uparrow\\hspace{-.2}\\right>$ initial state', fontsize = 10)
    # ax.grid()
    ax.legend()
    plt.tight_layout()
    plt.savefig(image_path)
    plt.show()

def plot_and_save_peff_Cso(singlet_phase, triplet_phase):
    so_scaling_values = (1e-4, 1e-3, 1e-2, 0.25, 0.5, 0.75, 1.0)

    array_paths = ( Path(__file__).parents[1] / 'data_produced' / 'arrays' / 'RbSr+_tcpld_so_scaling' / '100_E' / f'{singlet_phase:.4f}_{triplet_phase:.4f}' / 'dLMax_4' / f'{so_scaling:.4f}_hpf.txt' for so_scaling in so_scaling_values )
    output_state_resolved_arrays = list( np.loadtxt(array_path) for array_path in array_paths )
    image_path = Path(__file__).parents[1] / 'plots' / 'probability_scaling' / 'so_scaling' / f'peff_{singlet_phase:.4f}_{triplet_phase:.4f}.png'
    image_path.parent.mkdir(parents = True, exist_ok = True)
    pmf_path = Path(__file__).parents[1] / 'data' / 'pmf' / 'N_pdf_logic_params_EMM_500uK.txt'

    pmf = np.loadtxt(pmf_path)

    ss_dominated_rates = np.fromiter( (array[4] for array in output_state_resolved_arrays), dtype = float )
    xx = np.logspace(-4, 0, 100)

    print(ss_dominated_rates)
    fig, ax = plt.subplots()

    ax.scatter(so_scaling_values, ss_dominated_rates, s = 4**2, color = 'k', marker = 'o', label = '$p_\mathrm{eff}(c_\mathrm{so}) / p_\mathrm{eff}(c_\mathrm{so} = 1)$')
    ax.plot(xx, effective_probability(p0(max(ss_dominated_rates), pmf_array = pmf) * xx**4, pmf_array = pmf), color = 'red', linewidth = 1, linestyle = '--', label = '$p_\mathrm{eff}$ for $c_\mathrm{so}^4$ scaling')
    ax.plot(xx, effective_probability(p0(max(ss_dominated_rates), pmf_array = pmf) * xx**3, pmf_array = pmf), color = 'orange', linewidth = 1, linestyle = '--', label = '$p_\mathrm{eff}$ for $c_\mathrm{so}^3$ scaling')

    ax.axhline(0.0895, color = 'k', linewidth = 1, linestyle = '--', label = 'experimental value')

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylim(0.1*min(ss_dominated_rates), 10*max(ss_dominated_rates))
    ax.set_xlim(0.1*min(so_scaling_values), 10*max(so_scaling_values))
    ax.set_xlabel('spin-orbit coupling scaling factor $c_\mathrm{so}$')
    ax.set_ylabel('$p_\mathrm{eff}$')
    ax.set_title('Effective probability of the hyperfine energy release for $\left|2,2\\right>\hspace{0.2}\left|\\hspace{-.2}\\uparrow\\hspace{-.2}\\right>$ initial state', fontsize = 10)
    # ax.grid()
    ax.legend()
    plt.tight_layout()
    plt.savefig(image_path)
    plt.show()


def main():
    plot_and_save_peff_Cso(0.04, 0.24)

if __name__ == '__main__':
    main()