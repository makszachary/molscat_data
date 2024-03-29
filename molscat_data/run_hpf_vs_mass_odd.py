from pathlib import Path
import os
import time
from multiprocessing import Pool
import subprocess
import shutil
import re
import argparse

import numpy as np
from sigfig import round

import itertools

from matplotlib import pyplot as plt
import matplotlib as mpl

from _molscat_data import quantum_numbers as qn
from _molscat_data.smatrix import SMatrixCollection
from _molscat_data.smatrix_compatibility_copy import SMatrixCollection as SMatrixCollectionV0
from _molscat_data.thermal_averaging import n_root_scale, n_root_distribution, n_root_iterator
from _molscat_data.utils import rate_fmfsms_vs_L_SE, rate_fmfsms_vs_L_multiprocessing, rate_fmfsms_vs_L, k_L_E_SE_not_parallel
from _molscat_data.scaling_old import parameter_from_semiclassical_phase, default_singlet_phase_function, default_triplet_phase_function, default_singlet_parameter_from_phase, default_triplet_parameter_from_phase
from _molscat_data.analytical import MonoAlkaliEnergy
from _molscat_data.utils import k_L_E_not_parallel, probability, k_L_E_parallel_odd
from _molscat_data.effective_probability import effective_probability
from _molscat_data.physical_constants import amu_to_au, red_mass_87Rb_84Sr_amu, red_mass_87Rb_88Sr_amu
from prepare_so_coupling import scale_so_and_write
from _molscat_data.visualize import PartialRateVsEnergy, RateVsMagneticField


# we want to calculate rates at T from 0.1 mK to 10 mK, so we need E_min = 0.8e-6 K and E_max = 80 mK
# 70 partial waves should be safe for momentum-transfer rates at E = 8e-2 K (45 should be enough for spin exchange)
# we probably cannot afford for more than 100 energy values and 100 phase differences in the grid (its ~2h of molscat and ~12h of python per one singlet, triplet phase combinations, making up to ~44 hours for 100 triplet phases and with 34 cores)

# but for T = 0.5 mK and E_max = 4 mK we need only about 25 partial waves for spin-exchange and maybe 35 for momentum-transfer

scratch_path = Path(os.path.expandvars('$SCRATCH'))

data_dir_path = Path(__file__).parents[1] / 'data'
pickles_dir_path = scratch_path / 'python' / 'molscat_data' / 'data_produced' / 'pickles'
pickles_dir_path.mkdir(parents=True, exist_ok=True)
arrays_dir_path = pickles_dir_path.parent / 'arrays'
arrays_dir_path.mkdir(parents=True, exist_ok=True)
plots_dir_path = scratch_path / 'python' / 'molscat_data' / 'plots'

def create_and_run(molscat_input_template_path: Path | str, reduced_mass: float, singlet_phase: float, triplet_phase: float, so_scaling: float, energy_tuple: tuple[float, ...], L_max: int = 2*29, spin_orbit_included = True,) -> tuple[float, float, float]:
    time_0 = time.perf_counter()

    # L_max = 2*29
    F_values = (2,4,)
    S_values = (8,10,)
    T_min = np.amin([ [abs(F-S-L), abs(S-F-L), abs(L-F-S)] for F in F_values for S in S_values for L in range(0, L_max+1, 2)] )
    T_max = max(F_values)+max(S_values)+L_max

    molscat_energy_array_str = str(energy_tuple).strip(')').strip('(')
    nenergies = len(energy_tuple)
    E_min = min(energy_tuple)
    E_max = max(energy_tuple)

    singlet_scaling = default_singlet_parameter_from_phase(singlet_phase)
    triplet_scaling = default_triplet_parameter_from_phase(triplet_phase)

    molscat_executable_path = Path.home().joinpath('molscat-RKHS-tcpld', 'molscat-exe', 'molscat-RKHS-tcpld')
    molscat_input_templates_dir_path = Path(__file__).parents[1].joinpath('molscat', 'input_templates')
    molscat_input_path = scratch_path.joinpath('molscat', 'inputs', molscat_input_template_path.parent.relative_to(molscat_input_templates_dir_path), f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E', f'{singlet_phase:.4f}_{triplet_phase:.4f}', f'{so_scaling:.4f}', f'{reduced_mass:.4f}_amu', f'{molscat_input_template_path.stem}').with_suffix('.input')
    molscat_output_path  = scratch_path.joinpath('molscat', 'outputs', molscat_input_template_path.parent.relative_to(molscat_input_templates_dir_path), f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E', f'{singlet_phase:.4f}_{triplet_phase:.4f}', f'{so_scaling:.4f}', f'{reduced_mass:.4f}_amu', f'{molscat_input_template_path.stem}').with_suffix('.output')
    molscat_input_path.parent.mkdir(parents = True, exist_ok = True)
    molscat_output_path.parent.mkdir(parents = True, exist_ok = True)
    
    singlet_potential_path = Path(__file__).parents[1] / 'molscat' / 'potentials' / 'singlet.dat'
    triplet_potential_path = Path(__file__).parents[1] / 'molscat' / 'potentials' / 'triplet.dat'

    if spin_orbit_included:
        original_so_path = Path(__file__).parents[1] / 'data' / 'so_coupling' / 'lambda_SO_a_SrRb+_MT_original.dat'
        scaled_so_path = scratch_path / 'molscat' / 'so_coupling' / molscat_input_template_path.parent.relative_to(molscat_input_templates_dir_path) / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{triplet_phase:.4f}' / molscat_input_template_path.stem / f'so_{so_scaling:.4f}_scaling_{reduced_mass:.4f}_amu.dat'
        scaled_so_path.parent.mkdir(parents = True, exist_ok = True)
        scale_so_and_write(input_path = original_so_path, output_path = scaled_so_path, scaling = so_scaling)

    with open(molscat_input_template_path, 'r') as molscat_template:
        input_content = molscat_template.read()
        input_content = re.sub("NENERGIES", str(int(nenergies)), input_content, flags = re.M)
        input_content = re.sub("ENERGYARRAY", molscat_energy_array_str, input_content, flags = re.M)
        input_content = re.sub("NJTOTMIN", str(int(T_min)), input_content, flags = re.M)
        input_content = re.sub("NJTOTMAX", str(int(T_max)), input_content, flags = re.M)
        input_content = re.sub("NLMAX", str(int(L_max/2)), input_content, flags = re.M)
        input_content = re.sub("SINGLETPATH", f'\"{singlet_potential_path}\"', input_content, flags = re.M)
        input_content = re.sub("TRIPLETPATH", f'\"{triplet_potential_path}\"', input_content, flags = re.M)
        if spin_orbit_included:
            input_content = re.sub("SOPATH", f'\"{scaled_so_path}\"', input_content, flags = re.M)
        input_content = re.sub("SINGLETSCALING", str(singlet_scaling), input_content, flags = re.M)
        input_content = re.sub("TRIPLETSCALING", str(triplet_scaling), input_content, flags = re.M)
        input_content = re.sub("SOSCALING", str(1.00), input_content, flags = re.M)
        input_content = re.sub("NREDUCEDMASS", str(reduced_mass), input_content, flags = re.M)

        with open(molscat_input_path, 'w') as molscat_input:
            molscat_input.write(input_content)
            molscat_input.truncate()

    molscat_command = f"{molscat_executable_path} < {molscat_input_path} > {molscat_output_path}"
    print(f"{molscat_input_path.name} run\nwith the reduced mass: {reduced_mass:.4f}")
    subprocess.run(molscat_command, shell = True)

    duration = time.perf_counter()-time_0
    
    return duration, molscat_input_path, molscat_output_path


def create_and_run_parallel(molscat_input_templates, reduced_masses, singlet_phase, triplet_phase, so_scaling_values, energy_tuple: tuple[float, ...], L_max: int = 2*29,) -> set:
    t0 = time.perf_counter()
    output_dirs = []
    spin_orbit_included = True
    if so_scaling_values == None:
        so_scaling_values = (0.0,)
        spin_orbit_included = False
    with Pool() as pool:
       arguments = ( (x, reduced_mass, singlet_phase, triplet_phase, so_scaling_value, energy_tuple, L_max, spin_orbit_included) for x, reduced_mass, so_scaling_value in itertools.product( molscat_input_templates, reduced_masses, so_scaling_values))
       results = pool.starmap(create_and_run, arguments)
    
       for duration, input_path, output_path in results:
           output_dirs.append( output_path.parent )
           print(f"It took {duration:.2f} s to create the molscat input: {input_path}, run molscat and generate the output: {output_path}.")
    t1 = time.perf_counter()
    print(f"The time of the calculations in molscat was {t1 - t0:.2f} s.")

    return np.unique(output_dirs)


def collect_and_pickle(molscat_output_directory_path: Path | str, singlet_phase, triplet_phase, spinOrbitParameter: float | tuple[float, ...], energy_tuple: tuple[float, ...] ) -> tuple[SMatrixCollection, float, Path, Path]:
    time_0 = time.perf_counter()
    molscat_out_dir = scratch_path.joinpath('molscat', 'outputs')

    singlet_parameter = default_singlet_parameter_from_phase(singlet_phase)
    triplet_parameter = default_triplet_parameter_from_phase(triplet_phase)
    s_matrix_collection = SMatrixCollection(singletParameter = (singlet_parameter,), tripletParameter = (triplet_parameter,), collisionEnergy = energy_tuple)
    
    for output_path in Path(molscat_output_directory_path).iterdir():
        s_matrix_collection.update_from_output(file_path = output_path, non_molscat_so_parameter = spinOrbitParameter)
    
    pickle_path = pickles_dir_path / molscat_output_directory_path.relative_to(molscat_out_dir)
    pickle_path = pickle_path.parent / (pickle_path.name + '.pickle')
    pickle_path.parent.mkdir(parents = True, exist_ok = True)

    s_matrix_collection.toPickle(pickle_path)

    duration = time.perf_counter()-time_0

    return s_matrix_collection, duration, molscat_output_directory_path, pickle_path


# def calculate_and_save_k_L_E_and_peff_not_parallel(pickle_path: Path | str, phases = None, dLMax: int = 4, temperatures = (5e-4,)):
#     ### LOAD S-MATRIX, CALCULATE THE EFFECTIVE PROBABILITIES AND WRITE THEM TO .TXT FILE ###
#     t4 = time.perf_counter()
#     s_matrix_collection = SMatrixCollection.fromPickle(pickle_path)
    
#     so_scaling = s_matrix_collection.spinOrbitParameter
#     reduced_mass_amu = s_matrix_collection.reducedMass[0]/amu_to_au
#     # if len(so_scaling) == 1: so_scaling = float(so_scaling)

#     pmf_path = Path(__file__).parents[1].joinpath('data', 'pmf', 'N_pdf_logic_params_EMM_500uK.txt')
#     pmf_array = np.loadtxt(pmf_path)

#     param_indices = { "singletParameter": (s_matrix_collection.singletParameter.index(default_singlet_parameter_from_phase(phases[0])),), "tripletParameter": (s_matrix_collection.tripletParameter.index( default_triplet_parameter_from_phase(phases[1]) ), ) } if phases is not None else None

#     F_out, F_in, S = 2, 4, 1
#     MF_out, MS_out, MF_in, MS_in = np.meshgrid(np.arange(-F_out, F_out+1, 2), np.arange(-S, S+1, 2), np.arange(-F_in, F_in+1, 2), S, indexing = 'ij')
#     arg_hpf_deexcitation = (s_matrix_collection, F_out, MF_out, S, MS_out, F_in, MF_in, S, MS_in, param_indices, dLMax)

#     F_out, F_in, S = 4, 4, 1
#     MF_out, MS_out, MF_in, MS_in = np.meshgrid(np.arange(-F_out, F_out+1, 2), -S, np.arange(-F_in, F_in+1, 2), S, indexing = 'ij')
#     arg_cold_spin_change_higher = (s_matrix_collection, F_out, MF_out, S, MS_out, F_in, MF_in, S, MS_in, param_indices, dLMax)

#     F_out, F_in, S = 2, 2, 1
#     MF_out, MS_out, MF_in, MS_in = np.meshgrid(np.arange(-F_out, F_out+1, 2), -S, np.arange(-F_in, F_in+1, 2), S, indexing = 'ij')
#     arg_cold_spin_change_lower = (s_matrix_collection, F_out, MF_out, S, MS_out, F_in, MF_in, S, MS_in, param_indices, dLMax)

#     args = [arg_hpf_deexcitation, arg_cold_spin_change_higher, arg_cold_spin_change_lower]
#     names = [f'hyperfine deexcitation for the |f = 2, m_f = {{-2, -1, 0, 1, 2}}> |m_s = 1/2> initial states', 
#              f'cold spin change for the |f = 2, m_f = {{-2, -1, 0, 1, 2}}> |m_s = 1/2> initial states',
#              f'cold spin change for the |f = 1, m_f = {{-1, 0, 1}}> |m_s = 1/2> initial states']
#     abbreviations = ['hpf', 'cold_higher', 'cold_lower']

#     for abbreviation, name, arg in zip(*map(reversed, (abbreviations, names, args) ) ) :
#         t = time.perf_counter()

#         txt_path = arrays_dir_path.joinpath(pickle_path.relative_to(pickles_dir_path)).with_suffix('')
#         output_state_res_txt_path = txt_path / 'probabilities' / f'out_state_res_{abbreviation}.txt'
#         p0_txt_path = txt_path / 'probabilities' / f'p0_{abbreviation}.txt'
#         txt_path = txt_path / 'probabilities' / f'{abbreviation}.txt'
#         txt_path.parent.mkdir(parents = True, exist_ok = True)

#         rate_array, momentum_transfer_rate_array = k_L_E_not_parallel(*arg)
#         quantum_numbers = [ np.full_like(arg[2], arg[i]) for i in range(1, 9) ]
#         for index in np.ndindex(arg[2].shape):
#             k_L_E_txt_path = arrays_dir_path.joinpath(pickle_path.relative_to(pickles_dir_path)).with_suffix('')
#             k_L_E_txt_path = k_L_E_txt_path / f'k_L_E' / f'{abbreviation}' / f'OUT_{quantum_numbers[0][index]}_{quantum_numbers[1][index]}_{quantum_numbers[2][index]}_{quantum_numbers[3][index]}_IN_{quantum_numbers[4][index]}_{quantum_numbers[5][index]}_{quantum_numbers[6][index]}_{quantum_numbers[7][index]}.txt'
#             k_L_E_txt_path.parent.mkdir(parents = True, exist_ok = True)
#             np.savetxt(k_L_E_txt_path, rate_array[index].squeeze(), fmt = '%.10e', header = f'The energy-dependent rates of |F={quantum_numbers[4][index]}, MF={quantum_numbers[5][index]}>|S={quantum_numbers[6][index]}, MS={quantum_numbers[7][index]}> -> |F={quantum_numbers[0][index]}, MF={quantum_numbers[1][index]}>|S={quantum_numbers[2][index]}, MS={quantum_numbers[3][index]}> collisions ({name}) for each partial wave.\nThe values of reduced mass: {np.array(s_matrix_collection.reducedMass)/amu_to_au} a.m.u.\nThe singlet, triplet semiclassical phases: {phases}. The scaling of the short-range part of lambda_SO: {so_scaling}.\nThe maximum change of L: +/-{dLMax}. Energy values:\n{list(s_matrix_collection.collisionEnergy)}')
#             k_m_E_txt_path = arrays_dir_path.joinpath(pickle_path.relative_to(pickles_dir_path)).with_suffix('')
#             k_m_E_txt_path = k_m_E_txt_path / f'k_m_L_E' / f'{abbreviation}' / f'OUT_{quantum_numbers[0][index]}_{quantum_numbers[1][index]}_{quantum_numbers[2][index]}_{quantum_numbers[3][index]}_IN_{quantum_numbers[4][index]}_{quantum_numbers[5][index]}_{quantum_numbers[6][index]}_{quantum_numbers[7][index]}.txt'
#             k_m_E_txt_path.parent.mkdir(parents = True, exist_ok = True)
#             np.savetxt(k_m_E_txt_path, momentum_transfer_rate_array[index].squeeze(), fmt = '%.10e', header = f'The energy-dependent momentum-transfer rates calculated for the |F=2, MF=-2>|S=1, MS=-1> state for each partial wave.\nThe values of reduced mass: {np.array(s_matrix_collection.reducedMass)/amu_to_au} a.m.u.\nThe singlet, triplet semiclassical phases: {phases}. The scaling of the short-range part of lambda_SO: {so_scaling}.\nThe maximum change of L: +/-{dLMax}. Energy values:\n{list(s_matrix_collection.collisionEnergy)}')

#         distribution_arrays = [np.fromiter(n_root_iterator(temperature = temperature, E_min = min(s_matrix_collection.collisionEnergy), E_max = max(s_matrix_collection.collisionEnergy), N = len(s_matrix_collection.collisionEnergy), n = 3), dtype = float) for temperature in temperatures]
#         average_rate_arrays = np.array( [s_matrix_collection.thermalAverage(rate_array.sum(axis=len(arg[2].shape)), distribution_array) for distribution_array in distribution_arrays ] )
#         average_momentum_transfer_arrays = np.array( [ s_matrix_collection.thermalAverage(momentum_transfer_rate_array.sum(axis=len(arg[2].shape)), distribution_array) for distribution_array in distribution_arrays ] )
#         probability_arrays = average_rate_arrays / average_momentum_transfer_arrays
#         output_state_resolved_probability_arrays = probability_arrays.squeeze()
#         probability_arrays = probability_arrays.sum(axis = (1, 2)).squeeze()
#         effective_probability_arrays = effective_probability(probability_arrays, pmf_array)

#         temperatures_str = np.array2string( np.array(temperatures),formatter={'float_kind':lambda x: '%.2e' % x} )
#         momentum_transfer_str = np.array2string(average_momentum_transfer_arrays.reshape(average_momentum_transfer_arrays.shape[0], -1)[:,0], formatter={'float_kind':lambda x: '%.4e' % x} )

#         print("------------------------------------------------------------------------")
#         print(f'The bare (output-state-resolved) probabilities p_0 of the {name} for {phases=}, {so_scaling=}, {reduced_mass_amu=}, temperatures: {temperatures_str} K are:')
#         print(output_state_resolved_probability_arrays, '\n')

#         print("------------------------------------------------------------------------")
#         print(f'The bare probabilities p_0 of the {name} for {phases=}, {so_scaling=}, {reduced_mass_amu=}, temperatures: {temperatures_str} K are:')
#         print(probability_arrays, '\n')

#         print(f'The effective probabilities p_eff of the {name} for {phases=}, {so_scaling=}, {reduced_mass_amu=}, temperatures: {temperatures_str} K are:')
#         print(effective_probability_arrays)
#         print("------------------------------------------------------------------------")

#         np.savetxt(output_state_res_txt_path, output_state_resolved_probability_arrays.reshape(-1, output_state_resolved_probability_arrays.shape[-1]), fmt = '%.10f', header = f'[Original shape: {output_state_resolved_probability_arrays.shape}]\nThe bare (output-state-resolved) probabilities of the {name}.\nThe values of reduced mass: {np.array(s_matrix_collection.reducedMass)/amu_to_au} a.m.u.\nThe singlet, triplet semiclassical phases: {phases}. The scaling of the short-range part of lambda_SO: {so_scaling}.\nThe maximum change of L: +/-{dLMax}.\nTemperatures: {temperatures_str} K.\nThe momentum-transfer rate: {momentum_transfer_str} cm**3/s.')
#         np.savetxt(p0_txt_path, probability_arrays, fmt = '%.10f', header = f'[Original shape: {probability_arrays.shape}]\nThe effective probabilities of the {name}.\nThe values of reduced mass: {np.array(s_matrix_collection.reducedMass)/amu_to_au} a.m.u.\nThe singlet, triplet semiclassical phases: {phases}. The scaling of the short-range part of lambda_SO: {so_scaling}.\nThe maximum change of L: +/-{dLMax}.\nTemperatures: {temperatures_str} K.\nThe corresponding momentum-transfer rates: {momentum_transfer_str} cm**3/s.')
#         np.savetxt(txt_path, effective_probability_arrays, fmt = '%.10f', header = f'[Original shape: {effective_probability_arrays.shape}]\nThe effective probabilities of the {name}.\nThe values of reduced mass: {np.array(s_matrix_collection.reducedMass)/amu_to_au} a.m.u.\nThe singlet, triplet semiclassical phases: {phases}. The scaling of the short-range part of lambda_SO: {so_scaling}.\nThe maximum change of L: +/-{dLMax}.\nTemperatures: {temperatures_str} K.\nThe corresponding momentum-transfer rates: {momentum_transfer_str} cm**3/s.')
        
#         duration = time.perf_counter() - t
#         print(f"It took {duration:.2f} s.")

#     shutil.make_archive(arrays_dir_path.joinpath(pickle_path.relative_to(pickles_dir_path)).parent, 'zip' , arrays_dir_path.joinpath(pickle_path.relative_to(pickles_dir_path)).parent)
#     [shutil.rmtree(arrays_dir_path.joinpath(pickle_path.relative_to(pickles_dir_path)).with_suffix('') / name, ignore_errors=True) for name in ('k_L_E', 'k_m_L_E') ]
#     return


def calculate_and_save_peff_parallel(pickle_path: Path | str, transfer_pickle_path: Path | str, phases = None, dLMax: int = 4, temperatures = (5e-4,)):
    ### LOAD S-MATRIX, CALCULATE THE EFFECTIVE PROBABILITIES AND WRITE THEM TO .TXT FILE ###
    t4 = time.perf_counter()
    s_matrix_collection = SMatrixCollection.fromPickle(pickle_path)
    l_max = int(max(key[0].L for s_matrix in s_matrix_collection.matrixCollection.values() for key in s_matrix.matrix.keys())/2)

    transfer_s_matrix_collection = SMatrixCollection.fromPickle(transfer_pickle_path)
    transfer_l_max = int(max(key[0].L for s_matrix in transfer_s_matrix_collection.matrixCollection.values() for key in s_matrix.matrix.keys())/2)    

    so_scaling = s_matrix_collection.spinOrbitParameter
    reduced_mass_amu = s_matrix_collection.reducedMass[0]/amu_to_au

    pmf_path = Path(__file__).parents[1].joinpath('data', 'pmf', 'N_pdf_logic_params_EMM_500uK.txt')
    pmf_array = np.loadtxt(pmf_path)

    param_indices = { "singletParameter": (s_matrix_collection.singletParameter.index(default_singlet_parameter_from_phase(phases[0])),), "tripletParameter": (s_matrix_collection.tripletParameter.index( default_triplet_parameter_from_phase(phases[1]) ), ) } if phases is not None else None

    momentum_transfer_rate = transfer_s_matrix_collection.getMomentumTransferRateCoefficientVsL(qn.LF1F2(None, None, F1 = 4, MF1 = 4, F2 = 10, MF2 = 10), unit = 'cm**3/s', param_indices = param_indices)

    F_out, F_in, S_out, S_in = 2, 4, 10, 10
    MF_out, MS_out, MF_in, MS_in = np.meshgrid(np.arange(-F_out, F_out+1, 2), np.arange(-S_out, S_out+1, 2), np.arange(-F_in, F_in+1, 2), np.arange(-S_in, S_in+1, 2), indexing = 'ij')
    arg_hpf_deexcitation = (s_matrix_collection, F_out, MF_out, S_out, MS_out, F_in, MF_in, S_in, MS_in, param_indices, dLMax)

    F_out, F_in, S_out, S_in = 2, 4, 8, 10
    MF_out, MS_out, MF_in, MS_in = np.meshgrid(np.arange(-F_out, F_out+1, 2), np.arange(-S_out, S_out+1, 2), np.arange(-F_in, F_in+1, 2), np.arange(-S_in, S_in+1, 2), indexing = 'ij')
    arg_hpf_excitation_exchange = (s_matrix_collection, F_out, MF_out, S_out, MS_out, F_in, MF_in, S_in, MS_in, param_indices, dLMax)

    args = [arg_hpf_deexcitation, arg_hpf_excitation_exchange]# arg_cold_spin_change_higher, arg_cold_spin_change_lower]
    names = [f'hyperfine deexcitation for the |f_a = 2, m_f_a = {{-2, -1, 0, 1, 2}}> |f_i = 5, m_f_i = {{-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5}}> initial states', 
             f'hyperfine excitation exchange for the |f_a = 2, m_f_a = {{-2, -1, 0, 1, 2}}> |f_i = 5, m_f_i = {{-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5}}> initial states', ]# 
             #f'cold spin change for the |f = 2, m_f = {{-2, -1, 0, 1, 2}}> |m_s = 1/2> initial states',
             #f'cold spin change for the |f = 1, m_f = {{-1, 0, 1}}> |m_s = 1/2> initial states']
    abbreviations = ['hpf', 'hpf_exchange']# 'cold_higher', 'cold_lower']

    for abbreviation, name, arg in zip(*map(reversed, (abbreviations, names, args) ) ) :
        t = time.perf_counter()

        txt_path = arrays_dir_path.joinpath(pickle_path.relative_to(pickles_dir_path)).with_suffix('')
        output_state_res_txt_path = txt_path / 'probabilities' / f'out_state_res_{abbreviation}.txt'
        p0_txt_path = txt_path / 'probabilities' / f'p0_{abbreviation}.txt'
        txt_path = txt_path / 'probabilities' / f'{abbreviation}.txt'
        txt_path.parent.mkdir(parents = True, exist_ok = True)

        rate_array, _ = k_L_E_parallel_odd(*arg)

        ### WE ARE NOT WRITING THE k_L_E ARRAYS (~11*(11+9)*5*3 = 3300 FILES ONLY FOR HPF RELAXATION PER ONE PARAMETER / REDMASS - 55x MORE THAN IN THE CASE OF EVEN SR+ ISOTOPES)

        # quantum_numbers = [ np.full_like(arg[2], arg[i]) for i in range(1, 9) ]
        # for index in np.ndindex(arg[2].shape):
        #     k_L_E_txt_path = arrays_dir_path.joinpath(pickle_path.relative_to(pickles_dir_path)).with_suffix('')
        #     k_L_E_txt_path = k_L_E_txt_path / f'k_L_E' / f'{abbreviation}' / f'OUT_{quantum_numbers[0][index]}_{quantum_numbers[1][index]}_{quantum_numbers[2][index]}_{quantum_numbers[3][index]}_IN_{quantum_numbers[4][index]}_{quantum_numbers[5][index]}_{quantum_numbers[6][index]}_{quantum_numbers[7][index]}.txt'
        #     k_L_E_txt_path.parent.mkdir(parents = True, exist_ok = True)
        #     np.savetxt(k_L_E_txt_path, rate_array[index].squeeze(), fmt = '%.10e', header = f'The energy-dependent rates of |F={quantum_numbers[4][index]}, MF={quantum_numbers[5][index]}>|S={quantum_numbers[6][index]}, MS={quantum_numbers[7][index]}> -> |F={quantum_numbers[0][index]}, MF={quantum_numbers[1][index]}>|S={quantum_numbers[2][index]}, MS={quantum_numbers[3][index]}> collisions ({name}) for each partial wave.\nThe values of reduced mass: {np.array(s_matrix_collection.reducedMass)/amu_to_au} a.m.u.\nThe singlet, triplet semiclassical phases: {phases}. The scaling of the short-range part of lambda_SO: {so_scaling}.\nThe maximum change of L: +/-{dLMax}. Energy values:\n{list(s_matrix_collection.collisionEnergy)}')
        #     k_m_E_txt_path = arrays_dir_path.joinpath(pickle_path.relative_to(pickles_dir_path)).with_suffix('')
        #     k_m_E_txt_path = k_m_E_txt_path / f'k_m_L_E' / f'{abbreviation}' / f'OUT_{quantum_numbers[0][index]}_{quantum_numbers[1][index]}_{quantum_numbers[2][index]}_{quantum_numbers[3][index]}_IN_{quantum_numbers[4][index]}_{quantum_numbers[5][index]}_{quantum_numbers[6][index]}_{quantum_numbers[7][index]}.txt'
        #     k_m_E_txt_path.parent.mkdir(parents = True, exist_ok = True)
        #     np.savetxt(k_m_E_txt_path, momentum_transfer_rate_array[index].squeeze(), fmt = '%.10e', header = f'The energy-dependent momentum-transfer rates calculated for the |F=2, MF=-2>|S=1, MS=-1> state for each partial wave.\nThe values of reduced mass: {np.array(s_matrix_collection.reducedMass)/amu_to_au} a.m.u.\nThe singlet, triplet semiclassical phases: {phases}. The scaling of the short-range part of lambda_SO: {so_scaling}.\nThe maximum change of L: +/-{dLMax}. Energy values:\n{list(s_matrix_collection.collisionEnergy)}')

        distribution_arrays = [np.fromiter(n_root_iterator(temperature = temperature, E_min = min(s_matrix_collection.collisionEnergy), E_max = max(s_matrix_collection.collisionEnergy), N = len(s_matrix_collection.collisionEnergy), n = 3), dtype = float) for temperature in temperatures]
        average_rate_arrays = np.array( [s_matrix_collection.thermalAverage(rate_array.sum(axis=len(arg[2].shape)), distribution_array) for distribution_array in distribution_arrays ] )
        # in this script, momentum_transfer_rate is calculated within the script, not k_L_E_parallel function, and has shape (L_max, nenergies)
        momentum_transfer_rate_array = np.full((*arg[2].shape, *momentum_transfer_rate.shape), momentum_transfer_rate)
        average_momentum_transfer_arrays = np.array( [ s_matrix_collection.thermalAverage(momentum_transfer_rate_array.sum(axis=len(arg[2].shape)), distribution_array) for distribution_array in distribution_arrays ] )
        probability_arrays = average_rate_arrays / average_momentum_transfer_arrays
        output_state_resolved_probability_arrays = probability_arrays.squeeze()
        probability_arrays = probability_arrays.sum(axis = (1, 2)).squeeze()
        effective_probability_arrays = effective_probability(probability_arrays, pmf_array)

        temperatures_str = np.array2string( np.array(temperatures),formatter={'float_kind':lambda x: '%.2e' % x} )
        momentum_transfer_str = np.array2string(average_momentum_transfer_arrays.reshape(average_momentum_transfer_arrays.shape[0], -1)[:,0], formatter={'float_kind':lambda x: '%.4e' % x} )

        print("------------------------------------------------------------------------")
        print(f'The bare (output-state-resolved) probabilities p_0 of the {name} for {phases=}, {so_scaling=}, {reduced_mass_amu=}, temperatures: {temperatures_str} K are:')
        print(output_state_resolved_probability_arrays, '\n')

        print("------------------------------------------------------------------------")
        print(f'The bare probabilities p_0 of the {name} for {phases=}, {so_scaling=}, {reduced_mass_amu=}, temperatures: {temperatures_str} K are:')
        print(probability_arrays, '\n')

        print(f'The effective probabilities p_eff of the {name} for {phases=}, {so_scaling=}, {reduced_mass_amu=}, temperatures: {temperatures_str} K are:')
        print(effective_probability_arrays)
        print("------------------------------------------------------------------------")
        
        np.savetxt(output_state_res_txt_path, output_state_resolved_probability_arrays.reshape(-1, output_state_resolved_probability_arrays.shape[-1]), fmt = '%.10f', header = f'[Original shape: {output_state_resolved_probability_arrays.shape}]\nThe bare (output-state-resolved) probabilities of the {name}.\nThe values of reduced mass: {np.array(s_matrix_collection.reducedMass)/amu_to_au} a.m.u.\nThe singlet, triplet semiclassical phases: {phases}. The scaling of the short-range part of lambda_SO: {so_scaling}.\nThe maximum L: {l_max}. The maximum change of L: +/-{dLMax}.\nTemperatures: {temperatures_str} K.\nThe momentum-transfer rate: {momentum_transfer_str} cm**3/s.\nThe maximum L for the momentum-transfer rates calculations: {transfer_l_max}.')
        np.savetxt(p0_txt_path, probability_arrays.reshape(-1, probability_arrays.shape[-1]), fmt = '%.10f', header = f'[Original shape: {probability_arrays.shape}]\nThe effective probabilities of the {name}.\nThe values of reduced mass: {np.array(s_matrix_collection.reducedMass)/amu_to_au} a.m.u.\nThe singlet, triplet semiclassical phases: {phases}. The scaling of the short-range part of lambda_SO: {so_scaling}.\nThe maximum L: {l_max}. The maximum change of L: +/-{dLMax}.\nTemperatures: {temperatures_str} K.\nThe corresponding momentum-transfer rates: {momentum_transfer_str} cm**3/s.\nThe maximum L for the momentum-transfer rates calculations: {transfer_l_max}.')
        np.savetxt(txt_path, effective_probability_arrays.reshape(-1, effective_probability_arrays.shape[-1]), fmt = '%.10f', header = f'[Original shape: {effective_probability_arrays.shape}]\nThe effective probabilities of the {name}.\nThe values of reduced mass: {np.array(s_matrix_collection.reducedMass)/amu_to_au} a.m.u.\nThe singlet, triplet semiclassical phases: {phases}. The scaling of the short-range part of lambda_SO: {so_scaling}.\nThe maximum L: {l_max}. The maximum change of L: +/-{dLMax}.\nTemperatures: {temperatures_str} K.\nThe corresponding momentum-transfer rates: {momentum_transfer_str} cm**3/s.\nThe maximum L for the momentum-transfer rates calculations: {transfer_l_max}.')
        
        duration = time.perf_counter() - t
        print(f"It took {duration:.2f} s.")


def main():
    parser_description = "This is a python script for running molscat, collecting and pickling S-matrices, and calculating effective probabilities."
    parser = argparse.ArgumentParser(description=parser_description)
    parser.add_argument("-s", "--singlet_phase", type = float, default = None, help = "The singlet semiclassical phase modulo pi in multiples of pi.")
    parser.add_argument("-t", "--triplet_phase", type = float, default = None, help = "The triplet semiclassical phase modulo pi in multiples of pi.")
    parser.add_argument("-d", "--phase_difference", type = float, default = None, help = "The singlet-triplet semiclassical phase difference modulo pi in multiples of pi.")
    parser.add_argument("--so_scaling", nargs='*', type = float, default = [0.325,], help = "Values of the SO scaling.")
    parser.add_argument("--mass_min", type = float, default = 42.48, help = "Minimum reduced mass for the grid (in a.m.u.).")
    parser.add_argument("--mass_max", type = float, default = 43.80, help = "Maximum reduced mass for the grid (in a.m.u.).")
    parser.add_argument("--dmass", type = float, default = 0.01, help = "Mass step (in a.m.u.).")
    parser.add_argument("--nenergies", type = int, default = 50, help = "Number of energy values in a grid.")
    parser.add_argument("--E_min", type = float, default = 4e-7, help = "Lowest energy value in the grid.")
    parser.add_argument("--E_max", type = float, default = 4e-3, help = "Highest energy value in the grid.")
    parser.add_argument("--n_grid", type = int, default = 3, help = "n parameter for the nth-root energy grid.")
    parser.add_argument("--L_max", type = int, default = 2*29, help = "Maximum doubled partial wave included.")
    parser.add_argument("-T", "--temperatures", nargs='*', type = float, default = None, help = "Temperature in the Maxwell-Boltzmann distributions (in kelvins).")
    parser.add_argument("--input_dir_name", type = str, default = 'RbSr+_tcpld_vs_mass_odd', help = "Name of the directory with the molscat inputs")
    parser.add_argument("--transfer_input_dir_name", type = str, default = 'RbSr+_tcpld_momentum_transfer_vs_mass_odd', help = "Name of the directory with the molscat inputs")
    parser.add_argument("--molscat", action = 'store_true', help = "Include calculations in molscat.")
    parser.add_argument("--pickle", action = 'store_true', help = "Include pickling of molscat output.")
    parser.add_argument("--calc", action = 'store_true', help = "Include calculating probabilities from pickle.")
    args = parser.parse_args()


    nenergies, E_min, E_max, n = args.nenergies, args.E_min, args.E_max, args.n_grid
    energy_tuple = tuple( round(n_root_scale(i, E_min, E_max, nenergies-1, n = n), sigfigs = 11) for i in range(nenergies) )

    args.singlet_phase = args.singlet_phase if args.singlet_phase is not None else default_singlet_phase_function(1.0)
    args.triplet_phase = args.triplet_phase if args.triplet_phase is not None else args.singlet_phase + args.phase_difference if args.phase_difference is not None else default_triplet_phase_function(1.0)

    phases = ((args.singlet_phase, args.triplet_phase),)
    singlet_phase = args.singlet_phase
    triplet_phase = args.triplet_phase

    so_scaling_values = list(set(args.so_scaling))

    # reduced_masses = np.linspace(args.mass_min, args.mass_max, args.nmasses)
    reduced_masses = np.arange(args.mass_min, args.mass_max+0.5*args.dmass, args.dmass)

    if args.temperatures is None:
        temperatures = list(np.logspace(-4, -3, 10))
        temperatures.append(5e-4)
        temperatures = np.array(sorted(temperatures))
    else:
        temperatures = np.array(args.temperatures)

    molscat_input_templates = Path(__file__).parents[1].joinpath('molscat', 'input_templates', args.input_dir_name).iterdir()
    molscat_transfer_input_templates = Path(__file__).parents[1].joinpath('molscat', 'input_templates', args.transfer_input_dir_name).iterdir()

    if not (args.molscat or args.pickle or args.calc ):
        args.molscat, args.pickle, args.calc = True, True, True

    # ### RUN MOLSCAT ###
    if args.molscat:
        output_dirs = create_and_run_parallel(molscat_input_templates, reduced_masses, singlet_phase, triplet_phase, so_scaling_values, energy_tuple = energy_tuple, L_max = args.L_max)
        _ = create_and_run_parallel(molscat_transfer_input_templates, reduced_masses, singlet_phase, triplet_phase, (0.0,), energy_tuple, 2*79,)

    ## COLLECT S-MATRIX AND PICKLE IT ####
    if args.pickle:
        for reduced_mass in reduced_masses:
            transfer_output_dir = scratch_path / 'molscat' / 'outputs' / args.transfer_input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{triplet_phase:.4f}' / f'{0.0:.4f}' / f'{reduced_mass:.4f}_amu'
            _, duration, output_dir, transfer_pickle_path = collect_and_pickle( transfer_output_dir, singlet_phase, triplet_phase, None, energy_tuple, )
            for so_scaling_value in so_scaling_values:
                output_dir = scratch_path / 'molscat' / 'outputs' / args.input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{triplet_phase:.4f}' / f'{so_scaling_value:.4f}' / f'{reduced_mass:.4f}_amu'
                s_matrix_collection, duration, output_dir, pickle_path = collect_and_pickle( output_dir, singlet_phase, triplet_phase, so_scaling_value, energy_tuple)
                print(f"The time of gathering the outputs from {output_dir} into SMatrix object and pickling SMatrix into the file: {pickle_path} was {duration:.2f} s.")

    if args.calc:
        t0 = time.perf_counter()
        pickle_paths = tuple( tuple( pickles_dir_path / args.input_dir_name /f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{triplet_phase:.4f}' / f'{so_scaling_value:.4f}' / f'{reduced_mass:.4f}_amu.pickle' for reduced_mass in reduced_masses) for so_scaling_value in so_scaling_values)
        transfer_pickle_paths = tuple( pickles_dir_path / args.transfer_input_dir_name /f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{triplet_phase:.4f}' / f'{0.0:.4f}' / f'{reduced_mass:.4f}_amu.pickle' for reduced_mass in reduced_masses)
        [[calculate_and_save_peff_parallel(pickle_path, transfer_pickle_path, phases[0], 4, temperatures) for pickle_path, transfer_pickle_path in zip(tpl, transfer_pickle_paths, strict = True)] for tpl in pickle_paths]
        print(f'The time of calculating all the probabilities for all singlet, triplet phases was {time.perf_counter()-t0:.2f} s.')

if __name__ == "__main__":
    main()