### The scaling of the SO term should be done outside molscat - we don't want to scale the pure spin-spin part...

from typing import Any
import subprocess
import os
from pathlib import Path, PurePath
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
from _molscat_data.visualize import BarplotWide, ProbabilityVersusSpinOrbit
from prepare_so_coupling import scale_so_and_write
from copy_run_plot_k_L_E import save_and_plot_k_L_E_spinspin


singlet_scaling_path = Path(__file__).parents[1].joinpath('data', 'scaling_old', 'singlet_vs_coeff.json')
triplet_scaling_path = Path(__file__).parents[1].joinpath('data', 'scaling_old', 'triplet_vs_coeff.json')

E_min, E_max, nenergies, n = 4e-8, 4e-2, 200, 3
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

def collect_and_pickle(molscat_output_directory_path: Path | str, phases, spinOrbitParameter: float | tuple[float, ...] ) -> tuple[SMatrixCollection, float, Path, Path]:

    time_0 = time.perf_counter()
    molscat_out_dir = scratch_path.joinpath('molscat', 'outputs')
    singlet_parameter = tuple( default_singlet_parameter_from_phase(phase[0]) for phase in sorted(phases, key = lambda phase: phase[0]))
    triplet_parameter = tuple( default_triplet_parameter_from_phase(phase[1]) for phase in sorted(phases, key = lambda phase: phase[0]))
    s_matrix_collection = SMatrixCollection(singletParameter = singlet_parameter, tripletParameter = triplet_parameter, collisionEnergy = energy_tuple)
    
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
    output_dirs = []
    with Pool() as pool:
       arguments = ( (x, *y, z) for x, y, z in itertools.product( molscat_input_templates, phases, so_scaling_values ))
       results = pool.starmap(create_and_run, arguments)
    
       for duration, input_path, output_path in results:
           output_dirs.append( output_path.parent )
           print(f"It took {duration:.2f} s to create the molscat input: {input_path}, run molscat and generate the output: {output_path}.")
    t1 = time.perf_counter()
    print(f"The time of the calculations in molscat was {t1 - t0:.2f} s.")

    return output_dirs

def calculate_and_save_the_peff_parallel(pickle_path: Path | str, phases = None, dLMax: int = 4):
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
        output_state_res_txt_path = txt_path.parent / ('out_state_res_' + txt_path.name + '_' + abbreviation + '.txt')
        txt_path = txt_path.parent / (txt_path.name + '_' + abbreviation + '.txt')
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

def plot_and_save_p0_Cso(so_scaling_values, singlet_phase, triplet_phase):
    array_paths = ( arrays_dir_path / 'RbSr+_tcpld_so_scaling' / f'{nenergies}_E' / f'{singlet_phase:.4f}_{triplet_phase:.4f}' / f'out_state_res_{so_scaling:.4f}_hpf.txt' for so_scaling in so_scaling_values )
    output_state_resolved_arrays = list( np.loadtxt(array_path).reshape(3,2,5) for array_path in array_paths )
    image_path = plots_dir_path / 'probability_scaling' / 'so_scaling' / f'p0_{singlet_phase:.4f}_{triplet_phase:.4f}.png'
    image_path.parent.mkdir(parents = True, exist_ok = True)
    image_path_for_paper = plots_dir_path / 'plots' / 'probability_scaling' / 'so_scaling_for_paper' / f'p0_{singlet_phase:.4f}_{triplet_phase:.4f}.png'
    image_path_for_paper.parent.mkdir(parents = True, exist_ok = True)
    image_path_for_paper_linlin = plots_dir_path / 'plots' / 'probability_scaling' / 'so_scaling_for_paper_linlin' / f'p0_{singlet_phase:.4f}_{triplet_phase:.4f}.png'
    image_path_for_paper_linlin.parent.mkdir(parents = True, exist_ok = True)
    pmf_path = plots_dir_path / 'data' / 'pmf' / 'N_pdf_logic_params_EMM_500uK.txt'
    pmf_array = np.loadtxt(pmf_path)

    p_eff_exp = 0.0895
    p_eff_exp_std = 0.0242
    p0_exp = p0(p_eff_exp, pmf_array=pmf_array)
    dpeff = 1e-3
    p0_exp_std = (p0(p_eff_exp+dpeff/2, pmf_array=pmf_array)-p0(p_eff_exp-dpeff/2, pmf_array=pmf_array))/dpeff * p_eff_exp_std
    ss_dominated_rates = np.fromiter( (array.sum(axis = (0,1))[4] for array in output_state_resolved_arrays), dtype = float )


    
    fig, ax = ProbabilityVersusSpinOrbit.plotBareProbability(so_scaling_values, ss_dominated_rates, p0_exp = p0_exp, p0_exp_std = p0_exp_std)
    fig.savefig(image_path)

    ax.get_legend().remove()
    ax.set_title('')
    fig.savefig(image_path_for_paper)

    ax.set_yscale('lin')
    ax.set_xscale('lin')
    fig.savefig(image_path_for_paper_linlin)

    plt.close()

def plot_and_save_peff_Cso(so_scaling_values, singlet_phase, triplet_phase):
    array_paths = ( arrays_dir_path / 'data_produced' / 'arrays' / 'RbSr+_tcpld_so_scaling' / f'{nenergies}_E' / f'{singlet_phase:.4f}_{triplet_phase:.4f}' / f'{so_scaling:.4f}_hpf.txt' for so_scaling in so_scaling_values )
    output_state_resolved_arrays = list( np.loadtxt(array_path) for array_path in array_paths )
    image_path = plots_dir_path / 'probability_scaling' / 'so_scaling' / f'peff_{singlet_phase:.4f}_{triplet_phase:.4f}.png'
    image_path.parent.mkdir(parents = True, exist_ok = True)
    image_path_for_paper = plots_dir_path/ 'probability_scaling' / 'so_scaling_for_paper' / f'peff_{singlet_phase:.4f}_{triplet_phase:.4f}.png'
    image_path_for_paper.parent.mkdir(parents = True, exist_ok = True)
    image_path_for_paper_linlin = plots_dir_path / 'plots' / 'probability_scaling' / 'so_scaling_for_paper_linlin' / f'peff_{singlet_phase:.4f}_{triplet_phase:.4f}.png'
    pmf_path = Path(__file__).parents[1] / 'data' / 'pmf' / 'N_pdf_logic_params_EMM_500uK.txt'
    pmf_array = np.loadtxt(pmf_path)

    p_eff_exp = 0.0895
    p_eff_exp_std = 0.0242
    ss_dominated_rates = np.fromiter( (array[4] for array in output_state_resolved_arrays), dtype = float )

    fig, ax = ProbabilityVersusSpinOrbit.plotEffectiveProbability(so_scaling_values, ss_dominated_rates, p_eff_exp=p_eff_exp, p_eff_exp_std=p_eff_exp_std, pmf_array = pmf_array)
    fig.savefig(image_path)

    ax.get_legend().remove()
    ax.set_title('')
    fig.savefig(image_path_for_paper)

    ax.set_yscale('lin')
    ax.set_xscale('lin')
    fig.savefig(image_path_for_paper_linlin)

    plt.close()


def main():
    parser_description = "This is a python script for running molscat, collecting and pickling S-matrices, and calculating effective probabilities."
    parser = argparse.ArgumentParser(description=parser_description)
    parser.add_argument("-s", "--singlet_phase", type = float, default = None, help = "The singlet semiclassical phase modulo pi in multiples of pi.")
    parser.add_argument("-t", "--triplet_phase", type = float, default = None, help = "The triplet semiclassical phase modulo pi in multiples of pi.")
    args = parser.parse_args()

    # number_of_parameters = 24
    # all_phases = np.linspace(0.00, 1.00, (number_of_parameters+2) )[1:-1]
    # SINGLETSCALING = [parameter_from_semiclassical_phase(phase, singlet_scaling_path, starting_points=[1.000,1.010]) for phase in all_phases]
    # TRIPLETSCALING = [parameter_from_semiclassical_phase(phase, triplet_scaling_path, starting_points=[1.000,0.996]) for phase in all_phases]
    # scaling_combinations = itertools.product(SINGLETSCALING, TRIPLETSCALING)

    molscat_input_templates = Path(__file__).parents[1].joinpath('molscat', 'input_templates', 'RbSr+_tcpld_so_scaling').iterdir()
    singlet_phase = default_singlet_phase_function(1.0) if args.singlet_phase is None else args.singlet_phase
    triplet_phase = default_triplet_phase_function(1.0) if args.triplet_phase is None else args.triplet_phase
    phases = ((singlet_phase, triplet_phase),)
    so_scaling_values = (0.1, 0.15, 0.2, 0.25, 0.3, 0.34, 0.36, 0.37, 0.38, 0.39, 0.40, 0.42, 0.46, 0.5, 1.00)

    ### RUN MOLSCAT ###
    output_dirs = create_and_run_parallel(molscat_input_templates, phases, so_scaling_values)

    ### COLLECT S-MATRIX AND PICKLE IT ####
    # output_dir = Path(__file__).parents[1].joinpath('molscat', 'outputs', 'RbSr+_tcpld', f'{nenergies}_E', f'{args.singlet_phase}_{args.triplet_phase}')
    pickle_paths = []
    output_dirs = tuple( scratch_path / 'molscat' / 'outputs' / 'RbSr+_tcpld_so_scaling' / f'{nenergies}_E' / f'{phases[0][0]:.4f}_{phases[0][1]:.4f}' / f'{so_scaling:.4f}' for so_scaling in so_scaling_values )
    for output_dir, so_scaling in zip(output_dirs, so_scaling_values):
        s_matrix_collection, duration, output_dir, pickle_path = collect_and_pickle( output_dir, phases, so_scaling )
        pickle_paths.append(pickle_path)
        print(f"The time of gathering the outputs from {output_dir} into SMatrix object and pickling SMatrix into the file: {pickle_path} was {duration:.2f} s.")

    ### LOAD S-MATRIX, CALCULATE THE EFFECTIVE PROBABILITIES AND WRITE THEM TO .TXT FILE ###
    # pickle_path = Path(__file__).parents[1].joinpath('data_produced', 'pickles', 'RbSr+_tcpld_100_E.pickle')
    # pickle_path = Path(__file__).parents[1].joinpath('data_produced', 'pickles', 'RbSr+_tcpld', '10_E', f'{args.singlet_phase}_{args.triplet_phase}.pickle')
    # pickle_paths = tuple( pickles_dir_path / 'RbSr+_tcpld_so_scaling' / f'{nenergies}_E' / f'{phases[0][0]:.4f}_{phases[0][1]:.4f}' / f'{so_scaling:.4f}.pickle' for so_scaling in so_scaling_values )
    
    for pickle_path in pickle_paths:
        save_and_plot_k_L_E_spinspin(pickle_path)
        calculate_and_save_the_peff_parallel(pickle_path, phases[0])

    plot_and_save_p0_Cso(so_scaling_values, singlet_phase, triplet_phase)
    plot_and_save_peff_Cso(so_scaling_values, singlet_phase, triplet_phase)

    ### Calculate k_L(E) for the cold spin change from |2,2,up> state
    
    # for pickle_path in pickle_paths:
    #     save_and_plot_k_L_E_spinspin(pickle_path)


if __name__ == '__main__':
    main()
