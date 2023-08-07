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
from prepare_so_coupling import scale_so_and_write


singlet_scaling_path = Path(__file__).parents[1].joinpath('data', 'scaling_old', 'singlet_vs_coeff.json')
triplet_scaling_path = Path(__file__).parents[1].joinpath('data', 'scaling_old', 'triplet_vs_coeff.json')

# 70 partial waves should be safe for momentum-transfer rates at E = 8e-2 K (45 should be enough for spin exchange)
E_min, E_max, nenergies, n = 8e-7, 8e-2, 200, 3
energy_tuple = tuple( round(n_root_scale(i, E_min, E_max, nenergies-1, n = n), sigfigs = 11) for i in range(nenergies) )
molscat_energy_array_str = str(energy_tuple).strip(')').strip('(')
scratch_path = Path(os.path.expandvars('$SCRATCH'))

data_dir_path = Path(__file__).parents[1] / 'data'
pickles_dir_path = scratch_path / 'python' / 'molscat_data' / 'data_produced' / 'pickles'
pickles_dir_path.mkdir(parents=True, exist_ok=True)
arrays_dir_path = pickles_dir_path.parent / 'arrays'
arrays_dir_path.mkdir(parents=True, exist_ok=True)
plots_dir_path = scratch_path / 'python' / 'molscat_data' / 'plots'


def plot_probability_vs_DPhi(singlet_phases, triplet_phases, so_scaling, singlet_phase_distinguished = None, triplet_phases_distinguished = None):
    array_paths_hot = [ [arrays_dir_path / 'RbSr+_tcpld_80mK' / f'{nenergies}_E' / f'{singlet_phase:.4f}_{triplet_phase:.4f}' / f'{so_scaling:.4f}_hpf.txt' for triplet_phase in triplet_phases] for singlet_phase in singlet_phases]
    array_paths_cold_higher = [  [arrays_dir_path / 'RbSr+_tcpld_80mK' / f'{nenergies}_E' / f'{singlet_phase:.4f}_{triplet_phase:.4f}' / f'{so_scaling:.4f}_cold_higher.txt' for triplet_phase in triplet_phases] for singlet_phase in singlet_phases]
    arrays_hot = np.array([ [np.loadtxt(array_path) for array_path in sublist] for sublist in array_paths_hot ])
    arrays_cold_higher = np.array( [ [np.loadtxt(array_path) for array_path in sublist] for sublist in array_paths_cold_higher ] )

    if singlet_phase_distinguished is not None and triplet_phases_distinguished is not None:
        array_paths_hot_distinguished = [arrays_dir_path / 'RbSr+_tcpld_80mK' / f'{nenergies}_E' / f'{singlet_phase_distinguished:.4f}_{triplet_phase:.4f}' / f'{so_scaling:.4f}_hpf.txt' for triplet_phase in triplet_phases_distinguished]
        array_paths_cold_higher_distinguished = [arrays_dir_path / 'RbSr+_tcpld_80mK' / f'{nenergies}_E' / f'{singlet_phase_distinguished:.4f}_{triplet_phase:.4f}' / f'{so_scaling:.4f}_cold_higher.txt' for triplet_phase in triplet_phases_distinguished]
        arrays_hot_distinguished = np.array( [np.loadtxt(array_path) for array_path in array_paths_hot ] )
        arrays_cold_higher_distinguished = np.array( [np.loadtxt(array_path) for array_path in array_paths_cold_higher ] )
    
    png_path = plots_dir_path / 'paper' / 'DPhi_fitting' / 'two_point_all_curves' / f'SE_peff_vs_DPhi.png'
    svg_path = png_path.with_suffix('.svg')
    png_path.parent.mkdir(parents = True, exist_ok = True)
    # pmf_path = plots_dir_path / 'data' / 'pmf' / 'N_pdf_logic_params_EMM_500uK.txt'
    # pmf_array = np.loadtxt(pmf_path)

    exp_hot = np.loadtxt(data_dir_path / 'exp_data' / 'single_ion_hpf.dat')
    exp_cold_higher = np.loadtxt(data_dir_path / 'exp_data' / 'single_ion_cold_higher.dat')
    # exp_cold_lower = np.loadtxt(data_dir_path / 'exp_data' / 'single_ion_cold_lower.dat')

    experiment = np.array( [ exp_hot[0,0], exp_cold_higher[0,0] ] )
    std = np.array( [ exp_hot[1,0], exp_cold_higher[1,0] ] )


    xx = np.transpose( (np.meshgrid(singlet_phases, triplet_phases)[1]-np.meshgrid(singlet_phases, triplet_phases)[0]) % 1 )
    theory_distinguished = np.array( [ arrays_hot_distinguished[:,0], arrays_cold_higher_distinguished[:,0] ] ).transpose()
    theory = np.moveaxis(np.array( [ arrays_hot[:,:,0], arrays_cold_higher[:,:,0] ] ), 0, -1) if (singlet_phase_distinguished is not None and triplet_phases_distinguished is not None) else theory_distinguished

    fig, ax, ax_chisq = ValuesVsModelParameters.plotPeffAndChiSquaredVsDPhi(xx, theory, experiment, std, theory_distinguished)
    fig.savefig(png_path)
    fig.savefig(svg_path)

def main():
    parser_description = "This is a python script for running molscat, collecting and pickling S-matrices, and calculating effective probabilities."
    parser = argparse.ArgumentParser(description=parser_description)
    parser.add_argument("-s", "--singlet_phase", type = float, default = None, help = "The singlet semiclassical phase modulo pi in multiples of pi.")
    parser.add_argument("-t", "--triplet_phase", type = float, default = None, help = "The triplet semiclassical phase modulo pi in multiples of pi.")
    parser.add_argument("-d", "--phase_step", type = float, default = None, help = "The triplet semiclassical phase modulo pi in multiples of pi.")
    args = parser.parse_args()

    # number_of_parameters = 24
    # all_phases = np.linspace(0.00, 1.00, (number_of_parameters+2) )[1:-1]
    # SINGLETSCALING = [parameter_from_semiclassical_phase(phase, singlet_scaling_path, starting_points=[1.000,1.010]) for phase in all_phases]
    # TRIPLETSCALING = [parameter_from_semiclassical_phase(phase, triplet_scaling_path, starting_points=[1.000,0.996]) for phase in all_phases]
    # scaling_combinations = itertools.product(SINGLETSCALING, TRIPLETSCALING)

    molscat_input_templates = Path(__file__).parents[1].joinpath('molscat', 'input_templates', 'RbSr+_tcpld_so_scaling').iterdir()
    singlet_phase = default_singlet_phase_function(1.0) if args.singlet_phase is None else args.singlet_phase
    if args.phase_step is None:
        triplet_phase = default_triplet_phase_function(1.0) if args.triplet_phase is None else args.triplet_phase
    else:
        triplet_phase = np.array([( args.singlet_phase + phase_difference ) % 1 for phase_difference in np.arange(0, 1., args.phase_step) if (args.singlet_phase + phase_difference ) % 1 != 0 ] ).round(decimals=2)
    phases = np.around(tuple(zip(singlet_phase, triplet_phase, strict = True)), decimals = 2)
    so_scaling_values = (0.375,)

    # pickle_paths = tuple( pickles_dir_path / 'RbSr+_tcpld_80mK' / f'{nenergies}_E' / f'{phases[0][0]:.4f}_{triplet_phase:.4f}' / f'{so_scaling:.4f}.pickle' for triplet_phase in phases[1] for so_scaling in so_scaling_values )
    singlet_phase_distinguished = 0.0450
    triplet_phases_distinguished = np.array([( singlet_phase_distinguished + phase_difference ) % 1 for phase_difference in np.arange(0, 1., args.phase_step) if (args.singlet_phase + phase_difference ) % 1 != 0 ] ).round(decimals=2)
    plot_probability_vs_DPhi(singlet_phase, triplet_phases = triplet_phase, so_scaling = so_scaling_values[0], singlet_phase_distinguished = singlet_phase_distinguished, triplet_phases_distinguished = triplet_phases_distinguished)


if __name__ == '__main__':
    main()