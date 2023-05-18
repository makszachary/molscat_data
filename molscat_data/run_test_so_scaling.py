### The scaling of the SO term should be done outside molscat - we don't want to scale the pure spin-spin part...

from typing import Any
import subprocess
import os
from pathlib import Path, PurePath
import re
import argparse

from multiprocessing import Pool
# from multiprocessing.dummy import Pool as ThreadPool 

import itertools

import numpy as np
from sigfig import round

import time

from _molscat_data.smatrix import SMatrixCollection
from _molscat_data.thermal_averaging import n_root_scale
from _molscat_data.scaling_old import parameter_from_semiclassical_phase, semiclassical_phase_function
from _molscat_data.effective_probability import effective_probability
from _molscat_data.physical_constants import amu_to_au
from _molscat_data.utils import probability
from prepare_so_coupling import scale_so_and_write
from copy_run_plot_k_L_E import save_and_plot_k_L_E_spinspin

singlet_scaling_path = Path(__file__).parents[1].joinpath('data', 'scaling_old', 'singlet_vs_coeff.json')
triplet_scaling_path = Path(__file__).parents[1].joinpath('data', 'scaling_old', 'triplet_vs_coeff.json')

E_min, E_max, nenergies, n = 4e-7, 4e-3, 2, 3
energy_tuple = tuple( round(n_root_scale(i, E_min, E_max, nenergies-1, n = n), sigfigs = 11) for i in range(nenergies) )
molscat_energy_array_str = str(energy_tuple).strip(')').strip('(')
scratch_path = Path(os.path.expandvars('$SCRATCH'))

pickles_dir_path = scratch_path / 'python' / 'molscat_data' / 'data_produced' / 'pickles'
pickles_dir_path.mkdir(parents=True, exist_ok=True)
arrays_dir_path = pickles_dir_path.parent / 'arrays'
arrays_dir_path.mkdir(parents=True, exist_ok=True)
plots_dir_path = scratch_path / 'python' / 'molscat_data' / 'plots'

def create_and_run(molscat_input_template_path: Path | str, singlet_phase: float, triplet_phase: float, so_scaling: float) -> tuple[float, float, float]:
    
    time_0 = time.perf_counter()

    singlet_scaling = parameter_from_semiclassical_phase(singlet_phase, singlet_scaling_path, starting_points=[1.000,1.010])
    triplet_scaling = parameter_from_semiclassical_phase(triplet_phase, triplet_scaling_path, starting_points=[1.000,0.996])

    molscat_executable_path = Path.home().joinpath('molscat-RKHS-tcpld', 'molscat-exe', 'molscat-RKHS-tcpld')
    molscat_input_templates_dir_path = Path(__file__).parents[1].joinpath('molscat', 'input_templates')
    molscat_input_path = Path(__file__).parents[1].joinpath('molscat', 'inputs', molscat_input_template_path.parent.relative_to(molscat_input_templates_dir_path), f'{nenergies}_E', f'{singlet_phase:.4f}_{triplet_phase:.4f}', f'{so_scaling:.4f}', molscat_input_template_path.stem).with_suffix('.input')
    molscat_output_path  = scratch_path.joinpath('molscat', 'outputs', molscat_input_template_path.parent.relative_to(molscat_input_templates_dir_path), f'{nenergies}_E', f'{singlet_phase:.4f}_{triplet_phase:.4f}', f'{so_scaling:.4f}', molscat_input_template_path.stem).with_suffix('.output')
    molscat_input_path.parent.mkdir(parents = True, exist_ok = True)
    molscat_output_path.parent.mkdir(parents = True, exist_ok = True)
    
    singlet_potential_path = Path(__file__).parents[1] / 'molscat' / 'potentials' / 'singlet.dat'
    triplet_potential_path = Path(__file__).parents[1] / 'molscat' / 'potentials' / 'triplet.dat'
    original_so_path = Path(__file__).parents[1] / 'data' / 'so_coupling' / 'lambda_SO_a_SrRb+_MT_original.dat'
    scaled_so_path = Path(__file__).parents[1] / 'molscat' / 'so_coupling' / molscat_input_template_path.parent.relative_to(molscat_input_templates_dir_path) / f'{singlet_phase:.4f}_{triplet_phase:.4f}' / molscat_input_template_path.stem / f'so_{so_scaling:.3f}_scaling.dat'
    scaled_so_path.parent.mkdir(parents = True, exist_ok = True)

    scale_so_and_write(input_path = original_so_path, output_path = scaled_so_path, scaling = so_scaling)

    with open(molscat_input_template_path, 'r') as molscat_template:
        input_content = molscat_template.read()
        input_content = re.sub("NENERGIES", str(int(nenergies)), input_content, flags = re.M)
        input_content = re.sub("ENERGYARRAY", molscat_energy_array_str, input_content, flags = re.M)
        input_content = re.sub("SINGLETPATH", f'\"{singlet_potential_path}\"', input_content, flags = re.M)
        input_content = re.sub("TRIPLETPATH", f'\"{triplet_potential_path}\"', input_content, flags = re.M)
        input_content = re.sub("SOPATH", f'\"{scaled_so_path}\"', input_content, flags = re.M)
        input_content = re.sub("SINGLETSCALING", str(singlet_scaling), input_content, flags = re.M)
        input_content = re.sub("TRIPLETSCALING", str(triplet_scaling), input_content, flags = re.M)
        input_content = re.sub("SOSCALING", str(1.00), input_content, flags = re.M)

        with open(molscat_input_path, 'w') as molscat_input:
            molscat_input.write(input_content)
            molscat_input.truncate()

    molscat_command = f"{molscat_executable_path} < {molscat_input_path} > {molscat_output_path}"
    print(f"{molscat_input_path.name} run\nwith lambda_SO scaling: {so_scaling:.4f}")
    subprocess.run(molscat_command, shell = True)

    duration = time.perf_counter()-time_0
    
    return duration, molscat_input_path, molscat_output_path

def collect_and_pickle(molscat_output_directory_path: Path | str, singletParameter: tuple[float, ...], tripletParameter: tuple[float, ...], spinOrbitParameter: float | tuple[float, ...] ) -> tuple[SMatrixCollection, float, Path, Path]:

    time_0 = time.perf_counter()
    molscat_out_dir = scratch_path.joinpath('molscat', 'outputs')
    s_matrix_collection = SMatrixCollection(singletParameter = singletParameter, tripletParameter = tripletParameter, collisionEnergy = energy_tuple)
    
    for output_path in Path(molscat_output_directory_path).iterdir():
        s_matrix_collection.update_from_output(file_path = output_path, non_molscat_so_parameter = spinOrbitParameter)
    
    pickle_path = pickles_dir_path / molscat_output_directory_path.relative_to(molscat_out_dir)
    pickle_path = pickle_path.parent / (pickle_path.name + '.pickle')
    pickle_path.parent.mkdir(parents = True, exist_ok = True)

    s_matrix_collection.toPickle(pickle_path)

    duration = time.perf_counter()-time_0

    return s_matrix_collection, duration, molscat_output_directory_path, pickle_path

def create_and_run_parallel(molscat_input_templates, phases, so_scaling_values) -> set:
    t0 = time.perf_counter()
    output_dirs = set()
    with Pool() as pool:
       arguments = ( (x, *y, z) for x, y, z in itertools.product( molscat_input_templates, phases, so_scaling_values ))
       results = pool.starmap(create_and_run, arguments)
    
       for duration, input_path, output_path in results:
           output_dirs.add( output_path.parent )
           print(f"It took {duration:.2f} s to create the molscat input: {input_path}, run molscat and generate the output: {output_path}.")
    t1 = time.perf_counter()
    print(f"The time of the calculations in molscat was {t1 - t0:.2f} s.")

    return output_dirs

def calculate_and_save_the_peff_parallel(pickle_path, phases = None, dLMax: int = 4):
    ### LOAD S-MATRIX, CALCULATE THE EFFECTIVE PROBABILITIES AND WRITE THEM TO .TXT FILE ###
    t4 = time.perf_counter()
    fs = semiclassical_phase_function(singlet_scaling_path)
    ft = semiclassical_phase_function(triplet_scaling_path)
    s_matrix_collection = SMatrixCollection.fromPickle(pickle_path)
    
    so_scaling = s_matrix_collection.spinOrbitParameter
    # if len(so_scaling) == 1: so_scaling = float(so_scaling)

    pmf_path = Path(__file__).parents[1].joinpath('data', 'pmf', 'N_pdf_logic_params_EMM_500uK.txt')
    pmf_array = np.loadtxt(pmf_path)

    param_indices = { "singletParameter": (s_matrix_collection.singletParameter.index(parameter_from_semiclassical_phase(phases[0], singlet_scaling_path, starting_points=[1.000,1.010])),), "tripletParameter": (s_matrix_collection.tripletParameter.index( parameter_from_semiclassical_phase(phases[1], triplet_scaling_path, starting_points=[1.000,0.996]) ), ) } if phases is not None else None

    F_out, F_in, S = 2, 4, 1
    MF_out, MS_out, MF_in, MS_in = np.meshgrid(np.arange(-F_out, F_out+1, 2), np.arange(-S, S+1, 2), np.arange(-F_in, F_in+1, 2), S, indexing = 'ij')
    arg_hpf_deexcitation = (s_matrix_collection, F_out, MF_out, S, MS_out, F_in, MF_in, S, MS_in, param_indices, dLMax)

    F_out, F_in, S = 4, 4, 1
    MF_out, MS_out, MF_in, MS_in = np.meshgrid(np.arange(-F_out, F_out+1, 2), -S, np.arange(-F_in, F_in+1, 2), S, indexing = 'ij')
    arg_cold_spin_change_higher = (s_matrix_collection, F_out, MF_out, S, MS_out, F_in, MF_in, S, MS_in, param_indices, dLMax)

    F_out, F_in, S = 2, 2, 1
    MF_out, MS_out, MF_in, MS_in = np.meshgrid(np.arange(-F_out, F_out+1, 2), -S, np.arange(-F_in, F_in+1, 2), S, indexing = 'ij')
    arg_cold_spin_change_lower = (s_matrix_collection, F_out, MF_out, S, MS_out, F_in, MF_in, S, MS_in, param_indices, dLMax)

    args = [arg_hpf_deexcitation, arg_cold_spin_change_higher, arg_cold_spin_change_lower]
    names = [f'hyperfine deexcitation for the |f = 2, m_f = {{-2, -1, 0, 1, 2}}> |m_s = 1/2> initial states', 
             f'cold spin change for the |f = 2, m_f = {{-2, -1, 0, 1, 2}}> |m_s = 1/2> initial states',
             f'cold spin change for the |f = 1, m_f = {{-1, 0, 1}}> |m_s = 1/2> initial states']
    abbreviations = ['hpf', 'cold_higher', 'cold_lower']

    for abbreviation, name, arg in zip(*map(reversed, (abbreviations, names, args) ) ) :
        t = time.perf_counter()

        txt_path = arrays_dir_path.joinpath(pickle_path.relative_to(pickles_dir_path)).with_suffix('')
        # so_scaling = txt_path.name
        output_state_res_txt_path = txt_path.parent / f"dLMax_{dLMax}" / ('out_state_res_' + txt_path.name + '_' + abbreviation + '.txt')
        txt_path = txt_path.parent / f"dLMax_{dLMax}" / (txt_path.name + '_' + abbreviation + '.txt')
        txt_path.parent.mkdir(parents = True, exist_ok = True)

        probability_array = probability(*arg)
        output_state_resolved_probability_array = probability_array.squeeze()
        probability_array = probability_array.sum(axis = (0, 1)).squeeze()
        effective_probability_array = effective_probability(probability_array, pmf_array)

        print("------------------------------------------------------------------------")
        print(f'The bare (output-state-resolved) probabilities p_0 of the {name} for {phases=}, {so_scaling=}, {dLMax=} are:')
        print(output_state_resolved_probability_array, '\n')

        print("------------------------------------------------------------------------")
        print(f'The bare probabilities p_0 of the {name} for {phases=}, {so_scaling=}, {dLMax=} are:')
        print(probability_array, '\n')

        print(f'The effective probabilities p_eff of the {name} for {phases=}, {so_scaling=}, {dLMax=} are:')
        print(effective_probability_array)
        print("------------------------------------------------------------------------")
        
        np.savetxt(output_state_res_txt_path, output_state_resolved_probability_array.reshape(output_state_resolved_probability_array.shape[0], -1), fmt = '%.10f', header = f'[Original shape: {output_state_resolved_probability_array.shape}]\nThe bare (output-state-resolved) probabilities of the {name}.\nThe values of reduced mass: {np.array(s_matrix_collection.reducedMass)/amu_to_au} a.m.u.\nThe singlet, triplet semiclassical phases: {phases}. The scaling of the short-range part of lambda_SO: {so_scaling}.\nThe maximum change of L: +/-{dLMax}.')
        np.savetxt(txt_path, effective_probability_array, fmt = '%.10f', header = f'The effective probabilities of the {name}.\nThe values of reduced mass: {np.array(s_matrix_collection.reducedMass)/amu_to_au} a.m.u.\nThe singlet, triplet semiclassical phases: {phases}. The scaling of the short-range part of lambda_SO: {so_scaling}.\nThe maximum change of L: +/-{dLMax}.')
        
        duration = time.perf_counter() - t
        print(f"It took {duration:.2f} s.")

def main():
    parser_description = "This is a python script for running molscat, collecting and pickling S-matrices, and calculating effective probabilities."
    parser = argparse.ArgumentParser(description=parser_description)
    parser.add_argument("-s", "--singlet_phase", type = float, default = 0.04, help = "The singlet semiclassical phase modulo pi in multiples of pi.")
    parser.add_argument("-t", "--triplet_phase", type = float, default = 0.24, help = "The triplet semiclassical phase modulo pi in multiples of pi.")
    parser.add_argument("--dLMax", type = int, default = 4, help = "The maximum change of the orbital angular momentum L durign the collision.")
    args = parser.parse_args()

    number_of_parameters = 24
    all_phases = np.linspace(0.00, 1.00, (number_of_parameters+2) )[1:-1]
    SINGLETSCALING = [parameter_from_semiclassical_phase(phase, singlet_scaling_path, starting_points=[1.000,1.010]) for phase in all_phases]
    TRIPLETSCALING = [parameter_from_semiclassical_phase(phase, triplet_scaling_path, starting_points=[1.000,0.996]) for phase in all_phases]
    # scaling_combinations = itertools.product(SINGLETSCALING, TRIPLETSCALING)

    molscat_input_templates = Path(__file__).parents[1].joinpath('molscat', 'input_templates', 'RbSr+_tcpld_so_scaling').iterdir()
    phases = ((args.singlet_phase, args.triplet_phase),)
    so_scaling_values = (1e-4, 1e-3, 1e-2, 0.1, 0.25, 0.5, 0.75, 1.00)

    ### RUN MOLSCAT ###
    output_dirs = create_and_run_parallel(molscat_input_templates, phases, so_scaling_values)

    ### COLLECT S-MATRIX AND PICKLE IT ####
    # output_dir = Path(__file__).parents[1].joinpath('molscat', 'outputs', 'RbSr+_tcpld', f'{nenergies}_E', f'{args.singlet_phase}_{args.triplet_phase}')
    pickle_paths = set()
    for output_dir, so_scaling in zip(output_dirs, so_scaling_values):
        s_matrix_collection, duration, output_dir, pickle_path = collect_and_pickle( output_dir, SINGLETSCALING, TRIPLETSCALING, so_scaling )
        pickle_paths.add(pickle_path)
        print(f"The time of gathering the outputs from {output_dir} into SMatrix object and pickling SMatrix into the file: {pickle_path} was {duration:.2f} s.")

    ### LOAD S-MATRIX, CALCULATE THE EFFECTIVE PROBABILITIES AND WRITE THEM TO .TXT FILE ###
    # pickle_path = Path(__file__).parents[1].joinpath('data_produced', 'pickles', 'RbSr+_tcpld_100_E.pickle')
    # pickle_path = Path(__file__).parents[1].joinpath('data_produced', 'pickles', 'RbSr+_tcpld', '10_E', f'{args.singlet_phase}_{args.triplet_phase}.pickle')
    
    for pickle_path in pickle_paths:
        calculate_and_save_the_peff_parallel(pickle_path, phases[0], dLMax = args.dLMax)

    ### Calculate k_L(E) for the cold spin change from |2,2,up> state
    for pickle_path in pickle_paths:
        save_and_plot_k_L_E_spinspin(pickle_path)


if __name__ == '__main__':
    main()
