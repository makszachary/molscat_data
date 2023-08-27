### The scaling of the SO term is done outside molscat - we don't want to scale the pure spin-spin part...

from typing import Any
import subprocess
import os
from pathlib import Path
import re
import argparse

from multiprocessing import Pool

import itertools

import numpy as np
from sigfig import round

from matplotlib import pyplot as plt

import time

from _molscat_data.smatrix import SMatrixCollection
from _molscat_data.thermal_averaging import n_root_scale
from _molscat_data.scaling_old import parameter_from_semiclassical_phase, semiclassical_phase_function, default_singlet_parameter_from_phase, default_triplet_parameter_from_phase, default_singlet_phase_function, default_triplet_phase_function
from _molscat_data.effective_probability import effective_probability, p0
from _molscat_data.physical_constants import amu_to_au
from _molscat_data.utils import probability
from _molscat_data.visualize import ValuesVsModelParameters
from _molscat_data.chi_squared import chi_squared
from prepare_so_coupling import scale_so_and_write


singlet_scaling_path = Path(__file__).parents[1].joinpath('data', 'scaling_old', 'singlet_vs_coeff.json')
triplet_scaling_path = Path(__file__).parents[1].joinpath('data', 'scaling_old', 'triplet_vs_coeff.json')

scratch_path = Path(os.path.expandvars('$SCRATCH'))

data_dir_path = Path(__file__).parents[1] / 'data'
pickles_dir_path = scratch_path / 'python' / 'molscat_data' / 'data_produced' / 'pickles'
pickles_dir_path.mkdir(parents=True, exist_ok=True)
arrays_dir_path = pickles_dir_path.parent / 'arrays'
arrays_dir_path.mkdir(parents=True, exist_ok=True)
plots_dir_path = scratch_path / 'python' / 'molscat_data' / 'plots'

# ## laptop-test:
# nenergies = 100
# arrays_dir_path = Path(__file__).parents[1] / 'data_produced' / 'arrays'
# plots_dir_path = Path(__file__).parents[1] / 'plots'


def plot_probability_vs_DPhi(singlet_phases: float | np.ndarray[float], phase_differences: float | np.ndarray[float], so_scaling: float, energy_tuple: tuple[float, ...], singlet_phase_distinguished: float = None, triplet_phases_distinguished: float = None, temperatures: tuple[float, ...] = (5e-4,), plot_temperature: float = 5e-4, input_dir_name: str = 'RbSr+_tcpld_80mK', hybrid = False):
    nenergies = len(energy_tuple)
    E_min = min(energy_tuple)
    E_max = max(energy_tuple)
    singlet_phases, phase_differences = np.array(singlet_phases), np.array(phase_differences)
    probabilities_dir_name = 'probabilities_hybrid' if hybrid else 'probabilities'

    array_paths_hot = [ [arrays_dir_path / input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{(singlet_phase+phase_difference)%1:.4f}' / f'{so_scaling:.4f}' / probabilities_dir_name / 'hpf.txt' if ( singlet_phase+phase_difference ) % 1 !=0 else None for phase_difference in phase_differences ] for singlet_phase in singlet_phases]
    array_paths_cold_higher = [  [arrays_dir_path / input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{(singlet_phase+phase_difference)%1:.4f}' / f'{so_scaling:.4f}' / probabilities_dir_name / 'cold_higher.txt' if ( singlet_phase+phase_difference ) % 1 !=0 else None for phase_difference in phase_differences] for singlet_phase in singlet_phases]
    # arrays_hot = np.array([ [np.loadtxt(array_path) if array_path is not None else np.full(5, np.nan) for array_path in sublist] for sublist in array_paths_hot ]).reshape(*(array_paths_hot.shape), len(temperatures), -1)
    # [ print( np.loadtxt(array_path).shape ) if array_path is not None else np.full((len(temperatures), 5), np.nan) for sublist in array_paths_hot for array_path in sublist ]
    arrays_hot = np.array([ [np.loadtxt(array_path) if (array_path is not None and array_path.is_file()) else np.full((len(temperatures), 5), np.nan) for array_path in sublist] for sublist in array_paths_hot ])
    ## proposition:
    # arrays_hot = np.array([ [np.loadtxt(array_path) if (array_path is not None and array_path.is_file()) else np.full((len(temperatures), 5), np.nan) for array_path in sublist] for sublist in array_paths_hot ])
    arrays_hot = arrays_hot.reshape(*arrays_hot.shape[0:2], len(temperatures), -1)
    arrays_cold_higher = np.array( [ [np.loadtxt(array_path) if (array_path is not None and array_path.is_file()) else np.full((len(temperatures), 5), np.nan) for array_path in sublist] for sublist in array_paths_cold_higher ] )
    ## proposition:
    # arrays_cold_higher = np.array( [ [np.loadtxt(array_path) if (array_path is not None and array_path.is_file()) else np.full((len(temperatures), 5), np.nan) for array_path in sublist] for sublist in array_paths_cold_higher ] )
    arrays_cold_higher = arrays_cold_higher.reshape(*arrays_cold_higher.shape[0:2], len(temperatures), -1)
    singlet_phases = np.full((len(phase_differences), len(singlet_phases)), singlet_phases).transpose()
    triplet_phases = singlet_phases+phase_differences

    if singlet_phase_distinguished is not None and triplet_phases_distinguished is not None:
        array_paths_hot_distinguished = [arrays_dir_path / input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase_distinguished:.4f}_{(singlet_phase_distinguished+phase_difference)%1:.4f}' / f'{so_scaling:.4f}' / probabilities_dir_name / 'hpf.txt' if ( singlet_phase_distinguished+phase_difference ) % 1 !=0 else None for phase_difference in phase_differences]
        array_paths_cold_higher_distinguished = [arrays_dir_path / input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase_distinguished:.4f}_{(singlet_phase_distinguished+phase_difference)%1:.4f}' / f'{so_scaling:.4f}' / probabilities_dir_name / 'cold_higher.txt' if ( singlet_phase_distinguished+phase_difference ) % 1 !=0 else None for phase_difference in phase_differences]
        arrays_hot_distinguished = np.array( [np.loadtxt(array_path) if (array_path is not None and array_path.is_file()) else np.full((len(temperatures), 5), np.nan) for array_path in array_paths_hot_distinguished ] )
        arrays_hot_distinguished = arrays_hot_distinguished.reshape(arrays_hot_distinguished.shape[0], len(temperatures), -1)
        arrays_cold_higher_distinguished = np.array( [np.loadtxt(array_path) if (array_path is not None and array_path.is_file()) else np.full((len(temperatures), 5), np.nan) for array_path in array_paths_cold_higher_distinguished ] )
        arrays_cold_higher_distinguished = arrays_cold_higher_distinguished.reshape(arrays_cold_higher_distinguished.shape[0], len(temperatures), -1)
    
    suffix = '_hybrid' if hybrid else ''

    png_path = plots_dir_path / 'paper' / 'DPhi_fitting' / 'all_singlet' / f'{input_dir_name}{suffix}' / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'SE_peff_vs_DPhi_{plot_temperature:.2e}K.png'
    svg_path = png_path.with_suffix('.svg')
    png_path.parent.mkdir(parents = True, exist_ok = True)
    # pmf_path = plots_dir_path / 'data' / 'pmf' / 'N_pdf_logic_params_EMM_500uK.txt'
    # pmf_array = np.loadtxt(pmf_path)

    exp_hot = np.loadtxt(data_dir_path / 'exp_data' / 'single_ion_hpf.dat')
    exp_cold_higher = np.loadtxt(data_dir_path / 'exp_data' / 'single_ion_cold_higher.dat')
    # exp_cold_lower = np.loadtxt(data_dir_path / 'exp_data' / 'single_ion_cold_lower.dat')

    experiment = np.array( [ exp_hot[0,0], exp_cold_higher[0,0] ] )
    std = np.array( [ exp_hot[1,0], exp_cold_higher[1,0] ] )


    # xx = (np.meshgrid(singlet_phases, triplet_phases)[1]-np.meshgrid(singlet_phases, triplet_phases)[0]) % 1
    xx = np.full((len(singlet_phases), len(phase_differences)), phase_differences).transpose()
    T_index = np.nonzero(temperatures == plot_temperature)[0][0]
    theory_distinguished = np.moveaxis(np.array( [[ arrays_hot_distinguished[:,T_index,0],], [arrays_cold_higher_distinguished[:,T_index,0], ]] ), 0, -1)
    theory = np.moveaxis(np.array( [ arrays_hot[:,:,T_index,0], arrays_cold_higher[:,:,T_index,0] ] ), 0, -1) if (singlet_phase_distinguished is not None and triplet_phases_distinguished is not None) else theory_distinguished
    

    fig, ax, ax_chisq = ValuesVsModelParameters.plotPeffAndChiSquaredVsDPhi(xx, theory, experiment, std, theory_distinguished)
    # lines = ax_chisq.lines
    # data = np.array([line.get_xydata() for line in lines])
    # print(f'{minima.shape=}')
    # print(minima[-1])
    # print(f'{plot_temperature=}')
    # try:
    #     # print(np.nanargmin(data[:,:,1], axis=1))
    #     print(np.nanmin(data[:,:,1], axis=1))
    # except ValueError:
    #     print("ValueError raised, passing.")
    ax.set_ylim(0,1)
    ax.xaxis.get_major_ticks()[1].label1.set_visible(False)
    ax_chisq.legend(fontsize = 30, loc = 'upper left')
    fig.savefig(png_path)
    fig.savefig(svg_path)
    plt.close()

    chi_sq_distinguished = chi_squared(theory_distinguished, experiment, std)
    print(f'{chi_sq_distinguished.shape=}')
    print(f'{chi_sq_distinguished=}')

    minindex = np.nanargmin(chi_sq_distinguished)
    print(minindex)
    print(xx[:,0][minindex])
    print(np.nanmin(chi_sq_distinguished))

def main():
    parser_description = "This is a python script for running molscat, collecting and pickling S-matrices, and calculating effective probabilities."
    parser = argparse.ArgumentParser(description=parser_description)
    parser.add_argument("-d", "--phase_step", type = float, default = None, help = "The phase step multiples of pi.")
    parser.add_argument("-s", "--singlet_phase", type = float, default = None, help = "The distinguished value of the singlet semiclassical phase modulo pi in multiples of pi.")
    # parser.add_argument("-t", "--triplet_phase", type = float, default = None, help = "The distinguished value of the triplet semiclassical phase modulo pi in multiples of pi.")
    parser.add_argument("--nenergies", type = int, default = 100, help = "Number of energy values in a grid.")
    parser.add_argument("--E_min", type = float, default = 8e-7, help = "Lowest energy value in the grid.")
    parser.add_argument("--E_max", type = float, default = 8e-2, help = "Highest energy value in the grid.")
    parser.add_argument("--n_grid", type = int, default = 3, help = "n parameter for the nth-root energy grid.")
    parser.add_argument("-T", "--temperatures", nargs='*', type = float, default = None, help = "Temperature in the Maxwell-Boltzmann distributions (in kelvins).")
    parser.add_argument("--input_dir_name", type = str, default = 'RbSr+_tcpld_80mK', help = "Name of the directory with the molscat inputs")
    parser.add_argument("--hybrid", action = 'store_true', help = "If enabled, the probabilities will be taken from 'probabilities_hybrid' directories.")
    args = parser.parse_args()

    nenergies, E_min, E_max, n = args.nenergies, args.E_min, args.E_max, args.n_grid
    energy_tuple = tuple( round(n_root_scale(i, E_min, E_max, nenergies-1, n = n), sigfigs = 11) for i in range(nenergies) )

    # molscat_input_templates = Path(__file__).parents[1].joinpath('molscat', 'input_templates', 'RbSr+_tcpld_80mK').iterdir()
    singlet_phases = np.array([default_singlet_phase_function(1.0),]) if args.phase_step is None else np.arange(args.phase_step, 1., args.phase_step).round(decimals=4)
    triplet_phases = np.array([default_triplet_phase_function(1.0),]) if args.phase_step is None else np.arange(args.phase_step, 1., args.phase_step).round(decimals=4)
    phase_differences = np.arange(0, 1.+args.phase_step, args.phase_step).round(decimals=4)
    so_scaling_values = (0.375,)

    if args.temperatures is None:
        temperatures = list(np.logspace(-4, -2, 20))
        temperatures.append(5e-4)
        temperatures = np.array(sorted(temperatures))
    else:
        temperatures = np.array(args.temperatures)

    # pickle_paths = tuple( pickles_dir_path / 'RbSr+_tcpld_80mK' / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{phases[0][0]:.4f}_{triplet_phase:.4f}' / f'{so_scaling:.4f}.pickle' for triplet_phase in phases[1] for so_scaling in so_scaling_values )
    # default (ab initio) singlet phase is 0.0450\pi and default triplet phase is 0.7179\pi
    singlet_phase_distinguished = singlet_phases[0] if args.phase_step is None else default_singlet_phase_function(1.0) if args.singlet_phase is None else args.singlet_phase
    triplet_phases_distinguished = triplet_phases if args.phase_step is None else np.array([( singlet_phase_distinguished + phase_difference ) % 1 for phase_difference in np.arange(0, 1., args.phase_step) if (singlet_phase_distinguished + phase_difference ) % 1 != 0. ] ).round(decimals=4)
    
    # # ## laptop-test:
    # print(f'{singlet_phases=}')
    # print(f'{triplet_phases=}')
    # print(f'{singlet_phase_distinguished=}')
    # print(f'{triplet_phases_distinguished=}')
    # # singlet_phases = [0.64,]
    # # triplet_phases = np.arange(0.72, 0.81, 0.08)
    # # singlet_phase_distinguished = 0.60 if args.singlet_phase is None else args.singlet_phase
    # # triplet_phases_distinguished = triplet_phases if args.phase_step is None else np.arange(0.72, 0.77, 0.04).round(decimals=2)
    # # so_scaling_values = (0.25,)
    
    

    [plot_probability_vs_DPhi(singlet_phases = singlet_phases, phase_differences = phase_differences, so_scaling = so_scaling_values[0], energy_tuple = energy_tuple, singlet_phase_distinguished = singlet_phase_distinguished, triplet_phases_distinguished = triplet_phases_distinguished, temperatures = temperatures, plot_temperature = temperature, input_dir_name = args.input_dir_name, hybrid = args.hybrid) for temperature in temperatures]


if __name__ == '__main__':
    main()