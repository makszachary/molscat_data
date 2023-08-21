### The scaling of the SO term is done outside molscat - we don't want to scale the pure spin-spin part...

from typing import Any
import subprocess
import os
from pathlib import Path
import shutil
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
from _molscat_data.thermal_averaging import n_root_iterator
from _molscat_data.utils import probability, probability_not_parallel, k_L_E_not_parallel, k_L_E_SE_not_parallel
from _molscat_data.visualize import ValuesVsModelParameters
from prepare_so_coupling import scale_so_and_write


singlet_scaling_path = Path(__file__).parents[1].joinpath('data', 'scaling_old', 'singlet_vs_coeff.json')
triplet_scaling_path = Path(__file__).parents[1].joinpath('data', 'scaling_old', 'triplet_vs_coeff.json')

# we want to calculate rates at T from 0.1 mK to 10 mK, so we need E_min = 0.8e-6 K and E_max = 80 mK
# 70 partial waves should be safe for momentum-transfer rates at E = 8e-2 K (45 should be enough for spin exchange)
# we probably cannot afford for more than 100 energy values and 100 phase differences in the grid (its ~2h of molscat and ~12h of python per one singlet, triplet phase combinations, making up to ~44 hours for 100 triplet phases and with 34 cores)

scratch_path = Path(os.path.expandvars('$SCRATCH'))

data_dir_path = Path(__file__).parents[1] / 'data'
pickles_dir_path = scratch_path / 'python' / 'molscat_data' / 'data_produced' / 'pickles'
pickles_dir_path.mkdir(parents=True, exist_ok=True)
arrays_dir_path = pickles_dir_path.parent / 'arrays'
arrays_dir_path.mkdir(parents=True, exist_ok=True)
plots_dir_path = scratch_path / 'python' / 'molscat_data' / 'plots'

def create_and_run(molscat_input_template_path: Path | str, singlet_phase: float, triplet_phase: float, energy_tuple: tuple[float, ...]) -> tuple[float, float, float]:
    
    time_0 = time.perf_counter()

    molscat_energy_array_str = str(energy_tuple).strip(')').strip('(')
    nenergies = len(energy_tuple)
    E_min = min(energy_tuple)
    E_max = max(energy_tuple)

    singlet_scaling = parameter_from_semiclassical_phase(singlet_phase, singlet_scaling_path, starting_points=[1.000,1.010])
    triplet_scaling = parameter_from_semiclassical_phase(triplet_phase, triplet_scaling_path, starting_points=[1.000,0.996])

    molscat_executable_path = Path.home().joinpath('molscat-RKHS-tcpld', 'molscat-exe', 'molscat-RKHS-tcpld')
    molscat_input_templates_dir_path = Path(__file__).parents[1].joinpath('molscat', 'input_templates')
    molscat_input_path = scratch_path.joinpath('molscat', 'inputs', molscat_input_template_path.parent.relative_to(molscat_input_templates_dir_path), f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E', f'{singlet_phase:.4f}_{triplet_phase:.4f}', molscat_input_template_path.stem).with_suffix('.input')
    molscat_output_path  = scratch_path.joinpath('molscat', 'outputs', molscat_input_template_path.parent.relative_to(molscat_input_templates_dir_path), f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E', f'{singlet_phase:.4f}_{triplet_phase:.4f}', molscat_input_template_path.stem).with_suffix('.output')
    molscat_input_path.parent.mkdir(parents = True, exist_ok = True)
    molscat_output_path.parent.mkdir(parents = True, exist_ok = True)
    
    singlet_potential_path = Path(__file__).parents[1] / 'molscat' / 'potentials' / 'singlet.dat'
    triplet_potential_path = Path(__file__).parents[1] / 'molscat' / 'potentials' / 'triplet.dat'

    with open(molscat_input_template_path, 'r') as molscat_template:
        input_content = molscat_template.read()
        input_content = re.sub("NENERGIES", str(int(nenergies)), input_content, flags = re.M)
        input_content = re.sub("ENERGYARRAY", molscat_energy_array_str, input_content, flags = re.M)
        input_content = re.sub("SINGLETPATH", f'\"{singlet_potential_path}\"', input_content, flags = re.M)
        input_content = re.sub("TRIPLETPATH", f'\"{triplet_potential_path}\"', input_content, flags = re.M)
        input_content = re.sub("SINGLETSCALING", str(singlet_scaling), input_content, flags = re.M)
        input_content = re.sub("TRIPLETSCALING", str(triplet_scaling), input_content, flags = re.M)

        with open(molscat_input_path, 'w') as molscat_input:
            molscat_input.write(input_content)
            molscat_input.truncate()

    molscat_command = f"{molscat_executable_path} < {molscat_input_path} > {molscat_output_path}"
    print(f"{molscat_input_path.name} run without spin-orbit and spin-spin terms")
    subprocess.run(molscat_command, shell = True)

    duration = time.perf_counter()-time_0
    
    return duration, molscat_input_path, molscat_output_path

def collect_and_pickle(molscat_output_directory_path: Path | str, phases, energy_tuple: tuple[float, ...] ) -> tuple[SMatrixCollection, float, Path, Path]:

    time_0 = time.perf_counter()
    molscat_out_dir = scratch_path.joinpath('molscat', 'outputs')

    singlet_parameter = tuple( np.unique( [ default_singlet_parameter_from_phase(phase[0]) for phase in sorted(phases, key = lambda phase: phase[0]) ] ) )
    triplet_parameter = tuple( np.unique( [ default_triplet_parameter_from_phase(phase[1]) for phase in sorted(phases, key = lambda phase: phase[1]) ] ) )
    s_matrix_collection = SMatrixCollection(singletParameter = singlet_parameter, tripletParameter = triplet_parameter, collisionEnergy = energy_tuple)
    
    for output_path in Path(molscat_output_directory_path).iterdir():
        s_matrix_collection.update_from_output(file_path = output_path)
    
    pickle_path = pickles_dir_path / molscat_output_directory_path.relative_to(molscat_out_dir)
    pickle_path = pickle_path.parent / (pickle_path.name + '.pickle')
    pickle_path.parent.mkdir(parents = True, exist_ok = True)

    s_matrix_collection.toPickle(pickle_path)

    duration = time.perf_counter()-time_0

    return s_matrix_collection, duration, molscat_output_directory_path, pickle_path

def create_and_run_parallel(molscat_input_templates, phases, energy_tuple: tuple[float, ...]) -> set:
    t0 = time.perf_counter()
    output_dirs = []
    with Pool() as pool:
       arguments = ( (x, *y, energy_tuple) for x, y in itertools.product( molscat_input_templates, phases ))
       results = pool.starmap(create_and_run, arguments)
    
       for duration, input_path, output_path in results:
           output_dirs.append( output_path.parent )
           print(f"It took {duration:.2f} s to create the molscat input: {input_path}, run molscat and generate the output: {output_path}.")
    t1 = time.perf_counter()
    print(f"The time of the calculations in molscat was {t1 - t0:.2f} s.")

    return np.unique(output_dirs)

def calculate_and_save_k_L_E_SE_and_peff_not_parallel(pickle_path: Path | str, phases = None, temperatures = (5e-4,)):
    ### LOAD S-MATRIX, CALCULATE THE EFFECTIVE PROBABILITIES AND WRITE THEM TO .TXT FILE ###
    t4 = time.perf_counter()
    s_matrix_collection = SMatrixCollection.fromPickle(pickle_path)

    pmf_path = Path(__file__).parents[1].joinpath('data', 'pmf', 'N_pdf_logic_params_EMM_500uK.txt')
    pmf_array = np.loadtxt(pmf_path)

    param_indices = { "singletParameter": (s_matrix_collection.singletParameter.index(parameter_from_semiclassical_phase(phases[0], singlet_scaling_path, starting_points=[1.000,1.010])),), "tripletParameter": (s_matrix_collection.tripletParameter.index( parameter_from_semiclassical_phase(phases[1], triplet_scaling_path, starting_points=[1.000,0.996]) ), ) } if phases is not None else None

    F_out, F_in, S = 2, 4, 1
    MF_out, MS_out, MF_in, MS_in = np.meshgrid(np.arange(-F_out, F_out+1, 2), np.arange(-S, S+1, 2), np.arange(-F_in, F_in+1, 2), S, indexing = 'ij')
    arg_hpf_deexcitation = (s_matrix_collection, F_out, MF_out, S, MS_out, F_in, MF_in, S, MS_in, param_indices)

    F_out, F_in, S = 4, 4, 1
    MF_out, MS_out, MF_in, MS_in = np.meshgrid(np.arange(-F_out, F_out+1, 2), -S, np.arange(-F_in, F_in+1, 2), S, indexing = 'ij')
    arg_cold_spin_change_higher = (s_matrix_collection, F_out, MF_out, S, MS_out, F_in, MF_in, S, MS_in, param_indices)

    F_out, F_in, S = 2, 2, 1
    MF_out, MS_out, MF_in, MS_in = np.meshgrid(np.arange(-F_out, F_out+1, 2), -S, np.arange(-F_in, F_in+1, 2), S, indexing = 'ij')
    ### TEST:
    MF_out, MS_out, MF_in, MS_in = np.meshgrid(np.arange(-F_out, -F_out+3, 2), -S, np.arange(-F_in, -F_in+3, 2), S, indexing = 'ij')
    arg_cold_spin_change_lower = (s_matrix_collection, F_out, MF_out, S, MS_out, F_in, MF_in, S, MS_in, param_indices)

    args = [arg_hpf_deexcitation, arg_cold_spin_change_higher, arg_cold_spin_change_lower]
    names = [f'hyperfine deexcitation for the |f = 2, m_f = {{-2, -1, 0, 1, 2}}> |m_s = 1/2> initial states', 
             f'cold spin change for the |f = 2, m_f = {{-2, -1, 0, 1, 2}}> |m_s = 1/2> initial states',
             f'cold spin change for the |f = 1, m_f = {{-1, 0, 1}}> |m_s = 1/2> initial states']
    abbreviations = ['hpf', 'cold_higher', 'cold_lower']

    for abbreviation, name, arg in zip(*map(reversed, (abbreviations, names, args) ) ) :
        t = time.perf_counter()

        txt_path = arrays_dir_path.joinpath(pickle_path.relative_to(pickles_dir_path)).with_suffix('')
        output_state_res_txt_path = txt_path / 'probabilities' / f'out_state_res_{abbreviation}.txt'
        txt_path = txt_path / 'probabilities' / f'{abbreviation}.txt'
        txt_path.parent.mkdir(parents = True, exist_ok = True)

        rate_array, momentum_transfer_rate_array = k_L_E_SE_not_parallel(*arg)
        # print(f'{momentum_transfer_rate_array=}')
        # print(f'{momentum_transfer_rate_array.shape=}')
        quantum_numbers = [ np.full_like(arg[2], arg[i]) for i in range(1, 9) ]
        for index in np.ndindex(arg[2].shape):
            k_L_E_txt_path = arrays_dir_path.joinpath(pickle_path.relative_to(pickles_dir_path)).with_suffix('')
            k_L_E_txt_path = k_L_E_txt_path / f'k_L_E' / f'{abbreviation}' / f'OUT_{quantum_numbers[0][index]}_{quantum_numbers[1][index]}_{quantum_numbers[2][index]}_{quantum_numbers[3][index]}_IN_{quantum_numbers[4][index]}_{quantum_numbers[5][index]}_{quantum_numbers[6][index]}_{quantum_numbers[7][index]}.txt'
            k_L_E_txt_path.parent.mkdir(parents = True, exist_ok = True)
            np.savetxt(k_L_E_txt_path, rate_array[index].squeeze(), fmt = '%.10e', header = f'The energy-dependent rates of |F={quantum_numbers[4][index]}, MF={quantum_numbers[5][index]}>|S={quantum_numbers[6][index]}, MS={quantum_numbers[7][index]}> -> |F={quantum_numbers[0][index]}, MF={quantum_numbers[1][index]}>|S={quantum_numbers[2][index]}, MS={quantum_numbers[3][index]}> collisions ({name}) for each partial wave.\nThe values of reduced mass: {np.array(s_matrix_collection.reducedMass)/amu_to_au} a.m.u.\nThe singlet, triplet semiclassical phases: {phases}. Spin-orbit and spin-spin terms are NOT included.\nEnergy values:\n{list(s_matrix_collection.collisionEnergy)}')
            k_m_E_txt_path = arrays_dir_path.joinpath(pickle_path.relative_to(pickles_dir_path)).with_suffix('')
            k_m_E_txt_path = k_m_E_txt_path / f'k_m_L_E' / f'{abbreviation}' / f'IN_{quantum_numbers[4][index]}_{quantum_numbers[5][index]}_{quantum_numbers[6][index]}_{quantum_numbers[7][index]}.txt'
            k_m_E_txt_path.parent.mkdir(parents = True, exist_ok = True)
            np.savetxt(k_m_E_txt_path, momentum_transfer_rate_array[index].squeeze(), fmt = '%.10e', header = f'The energy-dependent momentum-transfer rates calculated for the |F=2, MF=-2>|S=1, MS=-1> state for each partial wave.\nThe values of reduced mass: {np.array(s_matrix_collection.reducedMass)/amu_to_au} a.m.u.\nThe singlet, triplet semiclassical phases: {phases}. Spin-orbit and spin-spin terms are NOT included.\nEnergy values:\n{list(s_matrix_collection.collisionEnergy)}')

        distribution_arrays = [np.fromiter(n_root_iterator(temperature = temperature, E_min = min(s_matrix_collection.collisionEnergy), E_max = max(s_matrix_collection.collisionEnergy), N = len(s_matrix_collection.collisionEnergy), n = 3), dtype = float) for temperature in temperatures]
        average_rate_arrays = np.array( [s_matrix_collection.thermalAverage(rate_array.sum(axis=len(arg[2].shape)), distribution_array) for distribution_array in distribution_arrays ] )
        average_momentum_transfer_arrays = np.array( [ s_matrix_collection.thermalAverage(momentum_transfer_rate_array.sum(axis=len(arg[2].shape)), distribution_array) for distribution_array in distribution_arrays ] )
        average_momentum_transfer_arrays = average_momentum_transfer_arrays[np.argwhere( quantum_numbers[4] == 4 & quantum_numbers[5] == 4 & quantum_numbers[6] == 1 & quantum_numbers[7] == 1 )]
        probability_arrays = average_rate_arrays / average_momentum_transfer_arrays
        output_state_resolved_probability_arrays = probability_arrays.squeeze()
        probability_arrays = probability_arrays.sum(axis = (1, 2)).squeeze()
        effective_probability_arrays = effective_probability(probability_arrays, pmf_array)

        temperatures_str = np.array2string( np.array(temperatures),formatter={'float_kind':lambda x: '%.2e' % x} )
        momentum_transfer_str = np.array2string(average_momentum_transfer_arrays.reshape(average_momentum_transfer_arrays.shape[0], -1)[:,0], formatter={'float_kind':lambda x: '%.4e' % x} )

        print("------------------------------------------------------------------------")
        print(f'The bare (output-state-resolved) probabilities p_0 of the {name} for {phases=} (spin-orbit and spin-spin terms NOT included), temperatures: {temperatures_str} K are:')
        print(output_state_resolved_probability_arrays, '\n')

        print("------------------------------------------------------------------------")
        print(f'The bare probabilities p_0 of the {name} for {phases=}, (spin-orbit and spin-spin terms NOT included), temperatures: {temperatures_str} K are:')
        print(probability_arrays, '\n')

        print(f'The effective probabilities p_eff of the {name} for {phases=}, (spin-orbit and spin-spin terms NOT included), temperatures: {temperatures_str} K are:')
        print(effective_probability_arrays)
        print("------------------------------------------------------------------------")

        np.savetxt(output_state_res_txt_path, output_state_resolved_probability_arrays.reshape(-1, output_state_resolved_probability_arrays.shape[-1]), fmt = '%.10f', header = f'[Original shape: {output_state_resolved_probability_arrays.shape}]\nThe bare (output-state-resolved) probabilities of the {name}.\nThe values of reduced mass: {np.array(s_matrix_collection.reducedMass)/amu_to_au} a.m.u.\nThe singlet, triplet semiclassical phases: {phases}. Spin-orbit and spin-spin terms are NOT included.\nTemperatures: {temperatures_str} K.\nThe momentum-transfer rate: {momentum_transfer_str} cm**3/s.')
        np.savetxt(txt_path, effective_probability_arrays, fmt = '%.10f', header = f'[Original shape: {effective_probability_arrays.shape}]\nThe effective probabilities of the {name}.\nThe values of reduced mass: {np.array(s_matrix_collection.reducedMass)/amu_to_au} a.m.u.\nThe singlet, triplet semiclassical phases: {phases}. Spin-orbit and spin-spin terms are NOT included.\nTemperatures: {temperatures_str} K.\nThe corresponding momentum-transfer rates: {momentum_transfer_str} cm**3/s.')
        
        duration = time.perf_counter() - t
        print(f"It took {duration:.2f} s.")

    shutil.make_archive(arrays_dir_path.joinpath(pickle_path.relative_to(pickles_dir_path)).with_suffix(''), 'zip' , arrays_dir_path.joinpath(pickle_path.relative_to(pickles_dir_path)).with_suffix(''))
    [shutil.rmtree(arrays_dir_path.joinpath(pickle_path.relative_to(pickles_dir_path)).with_suffix('') / name, ignore_errors=True) for name in ('k_L_E', 'k_m_L_E') ]
    return

def plot_probability_vs_DPhi(singlet_phase, triplet_phases, energy_tuple, temperatures, input_dir_name, plot_temperature = 5e-4):
    nenergies = len(energy_tuple)
    E_min = min(energy_tuple)
    E_max = max(energy_tuple)

    array_paths_hot = [ arrays_dir_path / input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{triplet_phase:.4f}' / 'probabilities' / f'hpf.txt' for triplet_phase in triplet_phases]
    array_paths_cold_higher = [ arrays_dir_path / input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{triplet_phase:.4f}' / 'probabilities' / f'cold_higher.txt' for triplet_phase in triplet_phases]
    arrays_hot = np.array([ np.loadtxt(array_path) for array_path in array_paths_hot ]).reshape(len(array_paths_hot), len(temperatures), -1)
    arrays_cold_higher = np.array( [np.loadtxt(array_path) for array_path in array_paths_cold_higher ] ).reshape(len(array_paths_cold_higher), len(temperatures), -1)

    png_path = plots_dir_path / 'paper' / 'DPhi_fitting' / 'one_singlet' / f'{input_dir_name}' / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}'  / f'SE_peff_vs_DPhi_{plot_temperature:.2e}K.png'
    svg_path = png_path.with_suffix('.svg')
    png_path.parent.mkdir(parents = True, exist_ok = True)
    # pmf_path = plots_dir_path / 'data' / 'pmf' / 'N_pdf_logic_params_EMM_500uK.txt'
    # pmf_array = np.loadtxt(pmf_path)

    exp_hot = np.loadtxt(data_dir_path / 'exp_data' / 'single_ion_hpf.dat')
    exp_cold_higher = np.loadtxt(data_dir_path / 'exp_data' / 'single_ion_cold_higher.dat')
    # exp_cold_lower = np.loadtxt(data_dir_path / 'exp_data' / 'single_ion_cold_lower.dat')

    experiment = np.array( [ exp_hot[0,0], exp_cold_higher[0,0] ] )
    std = np.array( [ exp_hot[1,0], exp_cold_higher[1,0] ] )

    xx = (np.array(triplet_phases) - singlet_phase) % 1
    T_index = np.nonzero(temperatures == plot_temperature)[0][0]
    theory_distinguished = np.array( [ arrays_hot[:,T_index,0], arrays_cold_higher[:,T_index,0] ] ).transpose()
    theory = theory_distinguished

    fig, ax, ax_chisq = ValuesVsModelParameters.plotPeffAndChiSquaredVsDPhi(xx, theory, experiment, std, theory_distinguished)
    ax.set_ylim(0,1)
    ax.xaxis.get_major_ticks()[1].label1.set_visible(False)
    ax_chisq.legend(fontsize = 30, loc = 'upper left')
    fig.savefig(png_path)
    fig.savefig(svg_path)

def calculate_peff_not_parallel_from_arrays(pickle_path: Path | str, momentum_pickle_path: Path | str, k_L_E_dir: Path | str, k_m_L_E_dir: Path | str, phases = None, temperatures = (5e-4,)):

    pmf_path = Path(__file__).parents[1].joinpath('data', 'pmf', 'N_pdf_logic_params_EMM_500uK.txt')
    pmf_array = np.loadtxt(pmf_path)

    s_matrix_collection = SMatrixCollection.fromPickle(pickle_path)
    momentum_s_matrix_collection = SMatrixCollection.fromPickle(momentum_pickle_path)

    pmf_path = Path(__file__).parents[1].joinpath('data', 'pmf', 'N_pdf_logic_params_EMM_500uK.txt')
    pmf_array = np.loadtxt(pmf_path)

    param_indices = { "singletParameter": (s_matrix_collection.singletParameter.index(default_singlet_parameter_from_phase(phases[0])),), "tripletParameter": (s_matrix_collection.tripletParameter.index( default_triplet_parameter_from_phase(phases[1]) ), ) } if phases is not None else None

    F_out, F_in, S = 2, 4, 1
    MF_out, MS_out, MF_in, MS_in = np.meshgrid(np.arange(-F_out, F_out+1, 2), np.arange(-S, S+1, 2), np.arange(-F_in, F_in+1, 2), S, indexing = 'ij')
    arg_hpf_deexcitation = (s_matrix_collection, F_out, MF_out, S, MS_out, F_in, MF_in, S, MS_in, param_indices)

    F_out, F_in, S = 4, 4, 1
    MF_out, MS_out, MF_in, MS_in = np.meshgrid(np.arange(-F_out, F_out+1, 2), -S, np.arange(-F_in, F_in+1, 2), S, indexing = 'ij')
    arg_cold_spin_change_higher = (s_matrix_collection, F_out, MF_out, S, MS_out, F_in, MF_in, S, MS_in, param_indices)

    F_out, F_in, S = 2, 2, 1
    MF_out, MS_out, MF_in, MS_in = np.meshgrid(np.arange(-F_out, F_out+1, 2), -S, np.arange(-F_in, F_in+1, 2), S, indexing = 'ij')
    ### TEST:
    MF_out, MS_out, MF_in, MS_in = np.meshgrid(np.arange(-F_out, -F_out+3, 2), -S, np.arange(-F_in, -F_in+3, 2), S, indexing = 'ij')
    arg_cold_spin_change_lower = (s_matrix_collection, F_out, MF_out, S, MS_out, F_in, MF_in, S, MS_in, param_indices)

    args = [arg_hpf_deexcitation, arg_cold_spin_change_higher, arg_cold_spin_change_lower]
    names = [f'hyperfine deexcitation for the |f = 2, m_f = {{-2, -1, 0, 1, 2}}> |m_s = 1/2> initial states', 
             f'cold spin change for the |f = 2, m_f = {{-2, -1, 0, 1, 2}}> |m_s = 1/2> initial states',
             f'cold spin change for the |f = 1, m_f = {{-1, 0, 1}}> |m_s = 1/2> initial states']
    abbreviations = ['hpf', 'cold_higher', 'cold_lower']

    for abbreviation, name, arg in zip(*map(reversed, (abbreviations, names, args) ) ) :
        t = time.perf_counter()
        
        quantum_numbers = [ np.full_like(arg[2], arg[i]) for i in range(1, 9) ]
        k_L_E_array_paths = [ k_L_E_dir / f'{abbreviation}' / f'OUT_{quantum_numbers[0][index]}_{quantum_numbers[1][index]}_{quantum_numbers[2][index]}_{quantum_numbers[3][index]}_IN_{quantum_numbers[4][index]}_{quantum_numbers[5][index]}_{quantum_numbers[6][index]}_{quantum_numbers[7][index]}.txt' for index in np.ndindex(arg[2].shape)]
        # k_m_L_E_array_paths = [ k_m_L_E_dir / f'{abbreviation}' / f'OUT_{quantum_numbers[0][index]}_{quantum_numbers[1][index]}_{quantum_numbers[2][index]}_{quantum_numbers[3][index]}_IN_{quantum_numbers[4][index]}_{quantum_numbers[5][index]}_{quantum_numbers[6][index]}_{quantum_numbers[7][index]}.txt' for index in np.ndindex(arg[2].shape)]
        k_m_L_E_array_paths = [ k_m_L_E_dir / f'hpf' / f'IN_4_4_1_1.txt' for index in np.ndindex(arg[2].shape)]

        rate_array = np.array([ k_L_E_array := np.loadtxt(k_L_E_array_path) for k_L_E_array_path in k_L_E_array_paths]).reshape((*(arg[2].shape), *(k_L_E_array.shape)))
        momentum_transfer_rate_array = np.array([ np.loadtxt(k_m_L_E_array_path) for k_m_L_E_array_path in k_m_L_E_array_paths]).reshape((*(arg[2].shape), *( np.loadtxt(k_m_L_E_array_paths[0]).shape)))

        # output_state_res_txt_path = k_L_E_dir.parent / 'probabilities_hybrid' / f'out_state_res_{abbreviation}.txt'
        # txt_path = k_L_E_dir.parent / 'probabilities_hybrid' / f'{abbreviation}.txt'
        # output_state_res_txt_path.parent.mkdir(parents = True, exist_ok = True)
        # txt_path.parent.mkdir(parents = True, exist_ok = True)

        txt_path = arrays_dir_path.joinpath(pickle_path.relative_to(pickles_dir_path)).with_suffix('')
        output_state_res_txt_path = txt_path / 'probabilities_hybrid' / f'out_state_res_{abbreviation}.txt'
        txt_path = txt_path / 'probabilities_hybrid' / f'{abbreviation}.txt'
        txt_path.parent.mkdir(parents = True, exist_ok = True)

        distribution_arrays = [np.fromiter(n_root_iterator(temperature = temperature, E_min = min(s_matrix_collection.collisionEnergy), E_max = max(s_matrix_collection.collisionEnergy), N = len(s_matrix_collection.collisionEnergy), n = 3), dtype = float) for temperature in temperatures]
        momentum_distribution_arrays = [np.fromiter(n_root_iterator(temperature = temperature, E_min = min(momentum_s_matrix_collection.collisionEnergy), E_max = max(momentum_s_matrix_collection.collisionEnergy), N = len(momentum_s_matrix_collection.collisionEnergy), n = 3), dtype = float) for temperature in temperatures]
        
        average_rate_arrays = np.array( [s_matrix_collection.thermalAverage(rate_array.sum(axis=len(arg[2].shape)), distribution_array) for distribution_array in distribution_arrays ] )
        average_momentum_transfer_arrays = np.array( [ momentum_s_matrix_collection.thermalAverage(momentum_transfer_rate_array.sum(axis=len(arg[2].shape)), momentum_distribution_array) for momentum_distribution_array in momentum_distribution_arrays ] )
        
        probability_arrays = average_rate_arrays / average_momentum_transfer_arrays
        output_state_resolved_probability_arrays = probability_arrays.squeeze()
        probability_arrays = probability_arrays.sum(axis = (1, 2)).squeeze()
        effective_probability_arrays = effective_probability(probability_arrays, pmf_array)

        temperatures_str = np.array2string( np.array(temperatures),formatter={'float_kind':lambda x: '%.2e' % x} )
        momentum_transfer_str = np.array2string(average_momentum_transfer_arrays.reshape(average_momentum_transfer_arrays.shape[0], -1)[:,0], formatter={'float_kind':lambda x: '%.4e' % x} )

        print("------------------------------------------------------------------------")
        print(f'The bare (output-state-resolved) probabilities p_0 of the {name} for {phases=} (spin-orbit and spin-spin terms NOT included), temperatures: {temperatures_str} K are:')
        print(output_state_resolved_probability_arrays, '\n')

        print("------------------------------------------------------------------------")
        print(f'The bare probabilities p_0 of the {name} for {phases=}, (spin-orbit and spin-spin terms NOT included), temperatures: {temperatures_str} K are:')
        print(probability_arrays, '\n')

        print(f'The effective probabilities p_eff of the {name} for {phases=}, (spin-orbit and spin-spin terms NOT included), temperatures: {temperatures_str} K are:')
        print(effective_probability_arrays)
        print("------------------------------------------------------------------------")

        np.savetxt(output_state_res_txt_path, output_state_resolved_probability_arrays.reshape(-1, output_state_resolved_probability_arrays.shape[-1]), fmt = '%.10f', header = f'[Original shape: {output_state_resolved_probability_arrays.shape}]\nThe bare (output-state-resolved) probabilities of the {name}.\nThe values of reduced mass: {np.array(s_matrix_collection.reducedMass)/amu_to_au} a.m.u.\nThe singlet, triplet semiclassical phases: {phases}. Spin-orbit and spin-spin terms are NOT included.\nTemperatures: {temperatures_str} K.\nThe momentum-transfer rate: {momentum_transfer_str} cm**3/s.')
        np.savetxt(txt_path, effective_probability_arrays, fmt = '%.10f', header = f'[Original shape: {effective_probability_arrays.shape}]\nThe effective probabilities of the {name}.\nThe values of reduced mass: {np.array(s_matrix_collection.reducedMass)/amu_to_au} a.m.u.\nThe singlet, triplet semiclassical phases: {phases}. Spin-orbit and spin-spin terms are NOT included.\nTemperatures: {temperatures_str} K.\nThe corresponding momentum-transfer rates: {momentum_transfer_str} cm**3/s.')
        
        duration = time.perf_counter() - t
        print(f"It took {duration:.2f} s.")

    return

def unpack_and_calculate_peff_from_arrays_and_remove(archive_paths: tuple[Path | str, ...], *args):
    archive_paths = tuple( Path(archive_path) for archive_path in archive_paths)
    unpack_paths = tuple( archive_path.parent / 'temp' / archive_path.with_suffix('').name for archive_path in archive_paths)
    print('A')
    [ shutil.unpack_archive(archive_path, unpack_path, 'zip') for archive_path, unpack_path in zip(archive_paths, unpack_paths) ]
    print('B')
    calculate_peff_not_parallel_from_arrays(*args)
    [ shutil.rmtree(unpack_path, ignore_errors=True) for unpack_path in unpack_paths ]
    


def main():
    parser_description = "This is a python script for running molscat, collecting and pickling S-matrices, and calculating effective probabilities."
    parser = argparse.ArgumentParser(description=parser_description)
    parser.add_argument("-s", "--singlet_phase", type = float, default = None, help = "The singlet semiclassical phase modulo pi in multiples of pi.")
    parser.add_argument("-t", "--triplet_phase", type = float, default = None, help = "The triplet semiclassical phase modulo pi in multiples of pi.")
    parser.add_argument("-d", "--phase_step", type = float, default = None, help = "The step of the phase difference in multiples of pi.")
    parser.add_argument("--nenergies", type = int, default = 100, help = "Number of energy values in a grid.")
    parser.add_argument("--E_min", type = float, default = 8e-7, help = "Lowest energy value in the grid.")
    parser.add_argument("--E_max", type = float, default = 8e-2, help = "Highest energy value in the grid.")
    parser.add_argument("--n_grid", type = int, default = 3, help = "n parameter for the nth-root energy grid.")
    parser.add_argument("-T", "--temperatures", nargs='*', type = float, default = None, help = "Temperature in the Maxwell-Boltzmann distributions (in kelvins).")
    parser.add_argument("--input_dir_name", type = str, default = 'RbSr+_tcpld_80mK', help = "Name of the directory with the molscat inputs")
    args = parser.parse_args()

    nenergies, E_min, E_max, n = args.nenergies, args.E_min, args.E_max, args.n_grid
    energy_tuple = tuple( round(n_root_scale(i, E_min, E_max, nenergies-1, n = n), sigfigs = 11) for i in range(nenergies) )

    molscat_input_templates = Path(__file__).parents[1].joinpath('molscat', 'input_templates', args.input_dir_name).iterdir()
    # singlet_phase = np.array([default_singlet_phase_function(1.0),]) if args.singlet_phase is None else np.array([args.singlet_phase,])
    singlet_phase = default_singlet_phase_function(1.0) if args.singlet_phase is None else args.singlet_phase
    if args.phase_step is None:
        triplet_phases = (default_triplet_phase_function(1.0),) if args.triplet_phase is None else args.triplet_phase
    else:
        triplet_phases = np.array([( singlet_phase + phase_difference ) % 1 for phase_difference in np.arange(0, 1., args.phase_step) if (singlet_phase + phase_difference ) % 1 != 0 ] ).round(decimals=4)
    phases = np.around(tuple((singlet_phase, triplet_phase) for triplet_phase in triplet_phases), decimals = 4)

    if args.temperatures is None:
        # temperatures = list(np.logspace(-4, -2, 20))
        temperatures = list(np.logspace(-3, -2, 10))
        temperatures.append(5e-4)
        temperatures = np.array(sorted(temperatures))
    else:
        temperatures = np.array(args.temperatures)
    

    ### RUN MOLSCAT ###
    #output_dirs = create_and_run_parallel(molscat_input_templates, phases, energy_tuple)

    # ### COLLECT S-MATRIX AND PICKLE IT ####
    # # output_dir = Path(__file__).parents[1].joinpath('molscat', 'outputs', 'RbSr+_tcpld', f'{nenergies}_E', f'{args.singlet_phase}_{args.triplet_phase}')
    pickle_paths = []
    for singlet_phase, triplet_phase in phases:
        output_dir = scratch_path / 'molscat' / 'outputs' / args.input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{triplet_phase:.4f}' 
        s_matrix_collection, duration, output_dir, pickle_path = collect_and_pickle( output_dir, ((singlet_phase, triplet_phase),), energy_tuple )
        pickle_paths.append(pickle_path)
        print(f"The time of gathering the outputs from {output_dir} into SMatrix object and pickling SMatrix into the file: {pickle_path} was {duration:.2f} s.")
    pickle_paths = np.unique(pickle_paths)

    ### LOAD S-MATRIX, CALCULATE THE EFFECTIVE PROBABILITIES AND WRITE THEM TO .TXT FILE ###
    # pickle_paths = tuple( pickles_dir_path / args.input_dir_name / f'{nenergies}_E' / f'{phases[0][0]:.4f}_{triplet_phase:.4f}.pickle' for triplet_phase in phases[1] )
    
    with Pool() as pool:
        t0 = time.perf_counter()
        pickle_paths = tuple( pickles_dir_path / args.input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{triplet_phase:.4f}.pickle' for singlet_phase, triplet_phase in phases)
        arguments = tuple( (pickle_path, phase, temperatures) for pickle_path, phase in zip(pickle_paths, phases) )
        pool.starmap(calculate_and_save_k_L_E_SE_and_peff_not_parallel, arguments)
        # [calculate_and_save_k_L_E_SE_and_peff_not_parallel(*arg) for arg in arguments]
        print(f'The time of calculating all the probabilities for all singlet, triplet phases was {time.perf_counter()-t0:.2f} s.')
    
    # [plot_probability_vs_DPhi(singlet_phase, triplet_phases = triplet_phases, energy_tuple = energy_tuple, temperatures = temperatures, input_dir_name = args.input_dir_name, plot_temperature=temperature) for temperature in temperatures]
    # _images_path = plots_dir_path / 'paper' / 'DPhi_fitting' / 'one_singlet' / f'{args.input_dir_name}' / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}'
    # shutil.make_archive(_images_path, 'zip' , _images_path)
    # shutil.rmtree(_images_path, ignore_errors=True)

    so_scaling = 0.375
    ####
    so_input_dir_name = args.input_dir_name[:args.input_dir_name.rfind('_SE')]
    ####
    archive_paths = [ (arrays_dir_path / so_input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{triplet_phase:.4f}.zip', arrays_dir_path / args.input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{triplet_phase:.4f}.zip') for singlet_phase, triplet_phase in phases ]
    
    pickle_paths = tuple( pickles_dir_path / so_input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{triplet_phase:.4f}' / f'{so_scaling:.4f}.pickle' for singlet_phase, triplet_phase in phases)
    momentum_pickle_paths = tuple( pickles_dir_path / args.input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{triplet_phase:.4f}.pickle' for singlet_phase, triplet_phase in phases)
    
    k_L_E_dirs = tuple( arrays_dir_path / so_input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / 'temp' / f'{singlet_phase:.4f}_{triplet_phase:.4f}' / f'{so_scaling:.4f}' / 'k_L_E' for singlet_phase, triplet_phase in phases)
    k_m_L_E_dirs = tuple( arrays_dir_path / args.input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / 'temp' / f'{singlet_phase:.4f}_{triplet_phase:.4f}' / 'k_m_L_E' for singlet_phase, triplet_phase in phases)

    with Pool() as pool:
        t0 = time.perf_counter()
        args = [ (archive_path, pickle_path, momentum_pickle_path, k_L_E_dir, k_m_L_E_dir, phase, temperatures) for archive_path, pickle_path, momentum_pickle_path, k_L_E_dir, k_m_L_E_dir, phase in zip(archive_paths, pickle_paths, momentum_pickle_paths, k_L_E_dirs, k_m_L_E_dirs, phases)]
        # pool.starmap(calculate_peff_not_parallel_from_arrays, args)
        pool.starmap(unpack_and_calculate_peff_from_arrays_and_remove, args)
        print(f'The time of calculating all the probabilities from k_L_E (so+ss) and k_m_L_E (w/o so+ss) arrays for all singlet, triplet phases was {time.perf_counter()-t0:.2f} s.')
    


if __name__ == '__main__':
    main()
