import subprocess
import os
import re

from pathlib import Path

import numpy as np

from matplotlib import pyplot as plt
from _molscat_data.effective_probability import effective_probability, p0
from _molscat_data.visualize import BarplotWide, ProbabilityVersusSpinOrbit


def plot_and_save_p0_Cso(singlet_phase, triplet_phase, relative_to_max = False):
    so_scaling_values = (1e-4, 1e-3, 1e-2, 0.25, 0.5, 0.75, 1.0)

    array_paths = ( Path(__file__).parents[1] / 'data_produced' / 'arrays' / 'RbSr+_tcpld_so_scaling' / '100_E' / f'{singlet_phase:.4f}_{triplet_phase:.4f}' / 'dLMax_4' / f'out_state_res_{so_scaling:.4f}_hpf.txt' for so_scaling in so_scaling_values )
    output_state_resolved_arrays = list( np.loadtxt(array_path).reshape(3,2,5) for array_path in array_paths )
    # image_path = Path(__file__).parents[1] / 'plots' / 'probability_scaling' / 'so_scaling' / f'{singlet_phase:.4f}_{triplet_phase:.4f}.png'
    image_path = Path(__file__).parents[1] / 'plots' / 'probability_scaling' / 'so_scaling' / f'p0_{singlet_phase:.4f}_{triplet_phase:.4f}.png'
    if relative_to_max:
        image_path = image_path.with_name(f'p0_rel_to_max_{singlet_phase:.4f}_{triplet_phase:.4f}.png')
    image_path.parent.mkdir(parents = True, exist_ok = True)

    ss_dominated_rates = np.fromiter( (array.sum(axis = (0,1))[4] for array in output_state_resolved_arrays), dtype = float )
    if relative_to_max:
        ss_dominated_rates = ss_dominated_rates / max(ss_dominated_rates)
    xx = np.logspace(-4, 1, 100)

    print(ss_dominated_rates)
    
    fig, ax = plt.subplots()
    
    data_label = '$p_0(c_\mathrm{so})'
    plot_title = 'Probability of the hyperfine energy release for $\left|2,2\\right>\hspace{0.2}\left|\\hspace{-.2}\\uparrow\\hspace{-.2}\\right>$ initial state'
    if relative_to_max:
        data_label = '$p_0(c_\mathrm{so}) / p_0(c_\mathrm{so} = 1)$'
        plot_title = 'Probability of the hyperfine energy release for $\left|2,2\\right>\hspace{0.2}\left|\\hspace{-.2}\\uparrow\\hspace{-.2}\\right>$ initial state\n(relative to the result for the original spin-orbit coupling from M.T.)'
    
    ax.scatter(so_scaling_values, ss_dominated_rates, s = 4**2, color = 'k', marker = 'o', label = data_label)
    ax.plot(xx, max(ss_dominated_rates) * xx**2, color = 'red', linewidth = 1, linestyle = '--', label = '$p_0 \sim c_\mathrm{so}^2$')
    ax.plot(xx, max(ss_dominated_rates) * xx**1.8, color = 'orange', linewidth = 1, linestyle = '--', label = '$p_0 \sim c_\mathrm{so}^{1.8}$')

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylim(0.1*min(ss_dominated_rates), 10*max(ss_dominated_rates))
    ax.set_xlim(0.1*min(so_scaling_values), 10*max(so_scaling_values))
    ax.set_xlabel('spin-orbit coupling scaling factor $c_\mathrm{so}$')
    ax.set_title(plot_title, fontsize = 10)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(image_path)
    plt.show()

def plot_and_save_peff_Cso(singlet_phase, triplet_phase):
    so_scaling_values = (1e-4, 1e-3, 1e-2, 0.25, 0.5, 0.75, 1.0)

    array_paths = ( Path(__file__).parents[1] / 'data_produced' / 'arrays' / 'RbSr+_tcpld_so_scaling' / '100_E' / f'{singlet_phase:.4f}_{triplet_phase:.4f}' / 'dLMax_4' / f'{so_scaling:.4f}_hpf.txt' for so_scaling in so_scaling_values )
    output_state_resolved_arrays = list( np.loadtxt(array_path) for array_path in array_paths )
    image_path = Path(__file__).parents[1] / 'plots' / 'probability_scaling' / 'so_scaling' / f'mod_peff_{singlet_phase:.4f}_{triplet_phase:.4f}.png'
    image_path.parent.mkdir(parents = True, exist_ok = True)
    pmf_path = Path(__file__).parents[1] / 'data' / 'pmf' / 'N_pdf_logic_params_EMM_500uK.txt'

    pmf = np.loadtxt(pmf_path)

    ss_dominated_rates = np.fromiter( (array[4] for array in output_state_resolved_arrays), dtype = float )
    xx = np.logspace(-4, 0, 100)

    print(ss_dominated_rates)
    fig, ax = plt.subplots()

    ax.scatter(so_scaling_values, ss_dominated_rates, s = 4**2, color = 'k', marker = 'o', label = '$p_\mathrm{eff}(c_\mathrm{so}) / p_\mathrm{eff}(c_\mathrm{so} = 1)$')
    ax.plot(xx, effective_probability(p0(max(ss_dominated_rates), pmf_array = pmf) * xx**2, pmf_array = pmf), color = 'red', linewidth = 1, linestyle = '--', label = '$p_\mathrm{eff}$ for $p_0 \sim c_\mathrm{so}^2$')
    ax.plot(xx, effective_probability(p0(max(ss_dominated_rates), pmf_array = pmf) * xx**1.8, pmf_array = pmf), color = 'orange', linewidth = 1, linestyle = '--', label = '$p_\mathrm{eff}$ for $p_0 \sim c_\mathrm{so}^{1.8}$')

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
    pmf_path = Path(__file__).parents[1] / 'data' / 'pmf' / 'N_pdf_logic_params_EMM_500uK.txt'
    pmf = np.loadtxt(pmf_path)
    print(effective_probability(0.122, pmf)+0.278*0.122)
    singlet_phase, triplet_phase = 0.04, 0.24
    so_scaling_values = (1e-4, 1e-3, 1e-2, 0.25, 0.5, 0.75, 1.0)

    array_paths = ( Path(__file__).parents[1] / 'data_produced' / 'arrays' / 'RbSr+_tcpld_so_scaling' / '100_E' / f'{singlet_phase:.4f}_{triplet_phase:.4f}' / 'dLMax_4' / f'out_state_res_{so_scaling:.4f}_hpf.txt' for so_scaling in so_scaling_values )
    output_state_resolved_arrays = list( np.loadtxt(array_path).reshape(3,2,5) for array_path in array_paths )
    ss_dominated_p0 = np.fromiter( (array.sum(axis = (0,1))[4] for array in output_state_resolved_arrays), dtype = float )

    image_path = Path(__file__).parents[1] / 'plots' / 'probability_scaling' / 'so_scaling' / f'p0_{singlet_phase:.4f}_{triplet_phase:.4f}.png'
    # if relative_to_max:
        # image_path = image_path.with_name(f'p0_rel_to_max_{singlet_phase:.4f}_{triplet_phase:.4f}.png')
    image_path.parent.mkdir(parents = True, exist_ok = True)

    fig1, ax1 = ProbabilityVersusSpinOrbit.plotBareProbability(so_parameter=so_scaling_values, probability=ss_dominated_p0, relative = True)
    # plt.show()
    
    array_paths = ( Path(__file__).parents[1] / 'data_produced' / 'arrays' / 'RbSr+_tcpld_so_scaling' / '100_E' / f'{singlet_phase:.4f}_{triplet_phase:.4f}' / 'dLMax_4' / f'{so_scaling:.4f}_hpf.txt' for so_scaling in so_scaling_values )
    output_state_resolved_arrays = list( np.loadtxt(array_path) for array_path in array_paths )
    ss_dominated_peff = np.fromiter( (array[4] for array in output_state_resolved_arrays), dtype = float )

    image_path = Path(__file__).parents[1] / 'plots' / 'probability_scaling' / 'so_scaling' / f'mod_peff_{singlet_phase:.4f}_{triplet_phase:.4f}.png'
    image_path.parent.mkdir(parents = True, exist_ok = True)    
    fig2, ax2 = ProbabilityVersusSpinOrbit.plotEffectiveProbability(so_parameter=so_scaling_values, probability = ss_dominated_peff)
    plt.show()

    # plot_and_save_p0_Cso(singlet_phase, triplet_phase)
    # plot_and_save_p0_Cso(singlet_phase, triplet_phase, relative_to_max=True)
    # plot_and_save_peff_Cso(singlet_phase, triplet_phase)

    so_scaling = 0.38
    SE_so_scaling = 0.01

    pmf_path = Path(__file__).parents[1].joinpath('data', 'pmf', 'N_pdf_logic_params_EMM_500uK.txt')
    pmf_array = np.loadtxt(pmf_path)

    exp_data_dir = Path(__file__).parents[1] / 'data' / 'exp_data'
    exp_hpf = exp_data_dir / 'single_ion_hpf.dat'
    exp_cold_higher = exp_data_dir / 'single_ion_cold_higher.dat'
    exp_cold_lower = exp_data_dir / 'single_ion_cold_lower.dat'

    theory_data_dir = Path(__file__).parents[1] / 'data_produced' / 'arrays' / 'RbSr+_tcpld_so_scaling' / '100_E' / f'{singlet_phase:.4f}_{triplet_phase:.4f}' / 'dLMax_4'
    theory_hpf = theory_data_dir / f'{so_scaling:.4f}_hpf.txt'
    theory_cold_higher = theory_data_dir / f'{so_scaling:.4f}_cold_higher.txt'
    theory_cold_lower = theory_data_dir / f'{so_scaling:.4f}_cold_lower.txt'
    SE_theory_hpf, SE_theory_cold_higher, SE_theory_cold_lower = ( theory_data_dir / f'{SE_so_scaling:.4f}_{name}.txt' for name in ( 'hpf', 'cold_higher', 'cold_lower' ) )
    
    # theory_data, exp_data, std_data = BarplotWide.prepareDataFromFiles(theory_hpf=theory_hpf, theory_cold_higher=theory_cold_higher, theory_cold_lower=theory_cold_lower,exp_hpf=exp_hpf,exp_cold_higher=exp_cold_higher, exp_cold_lower= exp_cold_lower)
    # SE_theory_data, _, __ = BarplotWide.prepareDataFromFiles(theory_hpf=SE_theory_hpf, theory_cold_higher=SE_theory_cold_higher, theory_cold_lower=SE_theory_cold_lower,exp_hpf=exp_hpf,exp_cold_higher=exp_cold_higher, exp_cold_lower= exp_cold_lower)
    # fig, ax1, ax2, ax3, legend_ax = BarplotWide.barplot(theory_data, exp_data, std_data, SE_theory_data = SE_theory_data)
    # BarplotWide.compareWithMatrixElements(fig, ax1, ax2, ax3, legend_ax, theory_data, pmf_array)
    
    # BarplotWide.addParams(fig, legend_ax, singlet_phase, triplet_phase, so_scaling)
    # # print(fig.axes)

    # image_path = Path(__file__).parents[1] / 'plots' / 'for MT' / f'{singlet_phase:.4f}_{triplet_phase:.4f}_{so_scaling:.4f}.png'
    # image_path.parent.mkdir(parents=True, exist_ok=True)
    # fig.savefig(image_path)
    # # plt.close()
    
    # plt.show()

    

if __name__ == '__main__':
    main()