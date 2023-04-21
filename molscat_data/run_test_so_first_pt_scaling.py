from typing import Any
import subprocess
from pathlib import Path, PurePath
import re
import argparse

from multiprocessing import Pool
# from multiprocessing.dummy import Pool as ThreadPool 

import itertools

import numpy as np
from sigfig import round

import time

from _molscat_data.smatrix import SMatrix, SMatrixCollection, CollectionParameters, CollectionParametersIndices
from _molscat_data import quantum_numbers as qn
from _molscat_data.thermal_averaging import n_root_scale
from old_utils.scaling_old import parameter_from_semiclassical_phase, semiclassical_phase_function
from _molscat_data.effective_probability import effective_probability
from _molscat_data.physical_constants import amu_to_au

singlet_scaling_path = Path(__file__).parents[1].joinpath('data', 'scaling_old', 'singlet_vs_coeff.json')
triplet_scaling_path = Path(__file__).parents[1].joinpath('data', 'scaling_old', 'triplet_vs_coeff.json')

E_min, E_max, nenergies, n = 4e-7, 4e-3, 100, 3
energy_tuple = tuple( round(n_root_scale(i, E_min, E_max, nenergies-1, n = n), sigfigs = 11) for i in range(nenergies) )
molscat_energy_array_str = str(energy_tuple).strip(')').strip('(')

def create_and_run(molscat_input_template_path: Path | str, singlet_phase: float, triplet_phase: float, first_point_scaling: float) -> tuple[float, float, float]:
    
    time_0 = time.perf_counter()

    lambda_so_template_path = Path(__file__).parents[1] / 'data' / 'so_coupling' / 'so_template_first_pt_scaling.dat'
    lambda_so_path = Path(__file__).parents[1] / 'molscat' / 'so_coupling' / f'so_{first_point_scaling:.2f}_first_pt_scaling.dat'
    lambda_so_path.parent.mkdir(parents = True, exist_ok = True)

    molscat_executable_path = Path.home().joinpath('molscat-RKHS-tcpld', 'molscat-exe', 'molscat-RKHS-tcpld')
    molscat_input_templates_dir_path = Path(__file__).parents[1].joinpath('molscat', 'input_templates')
    molscat_input_path = Path(__file__).parents[1].joinpath('molscat', 'inputs', molscat_input_template_path.parent.relative_to(molscat_input_templates_dir_path), f'{nenergies}_E', f'{singlet_phase:.2f}_{triplet_phase:.2f}', f'{first_point_scaling:.2f}', molscat_input_template_path.stem).with_suffix('.input')
    molscat_output_path  = Path(__file__).parents[1].joinpath('molscat', 'outputs', molscat_input_template_path.parent.relative_to(molscat_input_templates_dir_path), f'{nenergies}_E', f'{singlet_phase:.2f}_{triplet_phase:.2f}', f'{first_point_scaling:.2f}', molscat_input_template_path.stem).with_suffix('.output')
    molscat_input_path.parent.mkdir(parents = True, exist_ok = True)
    molscat_output_path.parent.mkdir(parents = True, exist_ok = True)
    
    singlet_scaling = parameter_from_semiclassical_phase(singlet_phase, singlet_scaling_path, starting_points=[1.000,1.010])
    triplet_scaling = parameter_from_semiclassical_phase(triplet_phase, triplet_scaling_path, starting_points=[1.000,0.996])

    with open(lambda_so_template_path, 'r') as lambda_so_template:
        lambda_so_content = lambda_so_template.read()
        lambda_so_content = re.sub("FIRSTPOINTVALUE", f'{first_point_scaling*4.435395152865600e-05:.15e}', lambda_so_content, flags = re.M)
        with open(lambda_so_path, 'w') as lambda_so_file:
            lambda_so_file.write(lambda_so_content)
            lambda_so_file.truncate()

    with open(molscat_input_template_path, 'r') as molscat_template:
        input_content = molscat_template.read()
        input_content = re.sub("NENERGIES", str(int(nenergies)), input_content, flags = re.M)
        input_content = re.sub("ENERGYARRAY", molscat_energy_array_str, input_content, flags = re.M)
        input_content = re.sub("SINGLETSCALING", str(singlet_scaling), input_content, flags = re.M)
        input_content = re.sub("TRIPLETSCALING", str(triplet_scaling), input_content, flags = re.M)
        input_content = re.sub("LAMBDASOFILEPATH", f'\"{lambda_so_path}\"', input_content, flags = re.M)

        with open(molscat_input_path, 'w') as molscat_input:
            molscat_input.write(input_content)
            molscat_input.truncate()

    molscat_command = f"{molscat_executable_path} < {molscat_input_path} > {molscat_output_path}"
    print(f"{molscat_input_path.name} run\nwith lambda_SO: {lambda_so_path}")
    subprocess.run(molscat_command, shell = True)

    duration = time.perf_counter()-time_0
    
    return duration, molscat_input_path, molscat_output_path

def collect_and_pickle(molscat_output_directory_path: Path | str, singletParameter: tuple[float, ...], tripletParameter: tuple[float, ...] ) -> tuple[SMatrixCollection, float, Path, Path]:

    time_0 = time.perf_counter()
    molscat_out_dir = Path(__file__).parents[1].joinpath('molscat', 'outputs')
    s_matrix_collection = SMatrixCollection(singletParameter = singletParameter, tripletParameter = tripletParameter, collisionEnergy = energy_tuple)
    
    for output_path in Path(molscat_output_directory_path).iterdir():
        s_matrix_collection.update_from_output(file_path = output_path)
    
    pickle_path = Path(__file__).parents[1].joinpath('data_produced', 'pickles', molscat_output_directory_path.relative_to(molscat_out_dir))
    pickle_path = pickle_path.parent / (pickle_path.name + '.pickle')
    pickle_path.parent.mkdir(parents = True, exist_ok = True)

    s_matrix_collection.toPickle(pickle_path)

    duration = time.perf_counter()-time_0

    return s_matrix_collection, duration, molscat_output_directory_path, pickle_path

def rate_fmfms_so(s_matrix_collection: SMatrixCollection, F_out: int, MF_out: int, MS_out: int, F_in: int, MF_in: int, MS_in: int, param_indices: dict) -> float:
    L_max = max(key[0].L for s_matrix in s_matrix_collection.matrixCollection.values() for key in s_matrix.matrix.keys())
    rate = np.sum( [ s_matrix_collection.getRateCoefficient(qn.LF1F2(L = L_out, ML = ML_in + MF_in + MS_in - MF_out - MS_out, F1 = F_out, MF1 = MF_out, F2 = 1, MF2 = MS_out), qn.LF1F2(L = L_in, ML = ML_in, F1 = F_in, MF1 = MF_in, F2 = 1, MF2 = MS_in), param_indices = param_indices) for L_in in range(0, L_max+1, 2) for ML_in in range(-L_in, L_in+1, 2) for L_out in range(L_in - 4, L_in + 5, 4) if L_out >= 0 ], axis = 0 )
    return rate

def probability(s_matrix_collection: SMatrixCollection, F_out: int | np.ndarray[Any, int], MF_out: int | np.ndarray[Any, int], MS_out: int | np.ndarray[Any, int], F_in: int | np.ndarray[Any, int], MF_in: int | np.ndarray[Any, int], MS_in: int | np.ndarray[Any, int], param_indices = None) -> np.ndarray[Any, float]:
    
    args = locals().copy()
    args.pop('s_matrix_collection')
    args.pop('param_indices')
    arg_shapes = tuple( value.shape for value in args.values() if isinstance(value, np.ndarray) )

    averaged_momentum_transfer_rate = s_matrix_collection.getThermallyAveragedMomentumTransferRate(qn.LF1F2(None, None, F1 = 2, MF1 = 2, F2 = 1, MF2 = -1), param_indices = param_indices)

    # convert all arguments to np.ndarrays if any of them is an instance np.ndarray
    array_like = False
    if any( isinstance(arg, np.ndarray) for arg in args.values() ):
        array_like = True
        arg_shapes = tuple( value.shape for value in args.values() if isinstance(value, np.ndarray) )
        if any(arg_shape != arg_shapes[0] for arg_shape in arg_shapes): raise ValueError(f"The shape of the numpy arrays passed as arguments should be the same.")
        
        for name, arg in args.items():
            if not isinstance(arg, np.ndarray):
                args[name] = np.full(arg_shapes[0], arg)


    if array_like:
        with Pool() as pool:
           arguments = ( (s_matrix_collection, *(args[name][index] for name in args), param_indices) for index in np.ndindex(arg_shapes[0]))
           results = pool.starmap(rate_fmfms_so, arguments)
           rate_shape = results[0].shape
           rate = np.array(results).reshape((*arg_shapes[0], *rate_shape))

           averaged_rate = s_matrix_collection.thermalAverage(rate)
           averaged_momentum_transfer_rate = np.full_like(averaged_rate, averaged_momentum_transfer_rate)
           probability = averaged_rate / averaged_momentum_transfer_rate

           return probability
    
    rate = rate_fmfms_so(s_matrix_collection, **args)
    averaged_rate = s_matrix_collection.thermalAverage(rate)
    probability = averaged_rate / averaged_momentum_transfer_rate

    return probability

def create_and_run_parallel(molscat_input_templates, phases, first_point_scaling_values) -> set:
    t0 = time.perf_counter()
    output_dirs = set()
    with Pool() as pool:
       arguments = ( (x, *y, z) for x, y, z in itertools.product( molscat_input_templates, phases, first_point_scaling_values ))
       results = pool.starmap(create_and_run, arguments)
    
       for duration, input_path, output_path in results:
           output_dirs.add( output_path.parent )
           print(f"It took {duration:.2f} s to create the molscat input: {input_path}, run molscat and generate the output: {output_path}.")
    t1 = time.perf_counter()
    print(f"The time of the calculations in molscat was {t1 - t0:.2f} s.")

    return output_dirs

def calculate_and_save_the_peff_parallel(pickle_path, phases = None, first_point_scaling = None):
    ### LOAD S-MATRIX, CALCULATE THE EFFECTIVE PROBABILITIES AND WRITE THEM TO .TXT FILE ###
    t4 = time.perf_counter()
    fs = semiclassical_phase_function(singlet_scaling_path)
    ft = semiclassical_phase_function(triplet_scaling_path)
    s_matrix_collection = SMatrixCollection.fromPickle(pickle_path)
    
    pmf_path = Path(__file__).parents[1].joinpath('data', 'pmf', 'N_pdf_logic_params_EMM_500uK.txt')
    pmf_array = np.loadtxt(pmf_path)

    param_indices = { "singletParameter": (s_matrix_collection.singletParameter.index(parameter_from_semiclassical_phase(phases[0], singlet_scaling_path, starting_points=[1.000,1.010])),), "tripletParameter": (s_matrix_collection.tripletParameter.index( parameter_from_semiclassical_phase(phases[1], triplet_scaling_path, starting_points=[1.000,0.996]) ), ) } if phases is not None else None

    F_out, F_in, S = 2, 4, 1
    MF_out, MS_out, MF_in, MS_in = np.meshgrid(np.arange(-F_out, F_out+1, 2), np.arange(-S, S+1, 2), np.arange(-F_in, F_in+1, 2), S, indexing = 'ij')
    arg_hpf_deexcitation = (s_matrix_collection, F_out, MF_out, MS_out, F_in, MF_in, MS_in, param_indices)

    F_out, F_in, S = 4, 4, 1
    MF_out, MS_out, MF_in, MS_in = np.meshgrid(np.arange(-F_out, F_out+1, 2), -S, np.arange(-F_in, F_in+1, 2), S, indexing = 'ij')
    arg_cold_spin_change_higher = (s_matrix_collection, F_out, MF_out, MS_out, F_in, MF_in, MS_in, param_indices)

    F_out, F_in, S = 2, 2, 1
    MF_out, MS_out, MF_in, MS_in = np.meshgrid(np.arange(-F_out, F_out+1, 2), -S, np.arange(-F_in, F_in+1, 2), S, indexing = 'ij')
    arg_cold_spin_change_lower = (s_matrix_collection, F_out, MF_out, MS_out, F_in, MF_in, MS_in, param_indices)

    args = [arg_hpf_deexcitation, arg_cold_spin_change_higher, arg_cold_spin_change_lower]
    names = [f'hyperfine deexcitation for the |f = 2, m_f = {{-2, -1, 0, 1, 2}}> |m_s = 1/2> initial states', 
             f'cold spin change for the |f = 2, m_f = {{-2, -1, 0, 1, 2}}> |m_s = 1/2> initial states',
             f'cold spin change for the |f = 1, m_f = {{-1, 0, 1}}> |m_s = 1/2> initial states']
    abbreviations = ['hpf', 'cold_higher', 'cold_lower']

    for abbreviation, name, arg in zip(*map(reversed, (abbreviations, names, args) ) ) :
        t = time.perf_counter()
        probability_array = probability(*arg).sum(axis = (0, 1)).squeeze()
        effective_probability_array = effective_probability(probability_array, pmf_array)

        print("------------------------------------------------------------------------")
        print(f'The bare probabilities p_0 of the {name} are:')
        print(probability_array, '\n')

        print(f'The effective probabilities p_eff of the {name} are:')
        print(effective_probability_array)
        print("------------------------------------------------------------------------")
        
        data_produced_dir = Path(__file__).parents[1].joinpath('data_produced')
        pickles_dir = data_produced_dir.joinpath('pickles')
        txt_dir = data_produced_dir.joinpath('arrays')
        txt_path = txt_dir.joinpath(pickle_path.relative_to(pickles_dir)).with_suffix('')
        txt_path = txt_path.parent / (txt_path.name + '_' + abbreviation + '.txt')
        # txt_path = pickle_path.parent.joinpath('arrays', pickle_path.stem+'_'+abbreviation).with_suffix('.txt')
        txt_path.parent.mkdir(parents = True, exist_ok = True)
        np.savetxt(txt_path, effective_probability_array, fmt = '%.10f', header = f'The effective probabilities of the {name}.\nThe values of reduced mass: {np.array(s_matrix_collection.reducedMass)/amu_to_au} a.m.u.\nThe singlet, triplet semiclassical phases: {phases}. The scaling of the first point in lambda_SO: {first_point_scaling}.')
        
        duration = time.perf_counter() - t
        print(f"It took {duration:.2f} s.")

def main():
    parser_description = "This is a python script for running molscat, collecting and pickling S-matrices, and calculating effective probabilities."
    parser = argparse.ArgumentParser(description=parser_description)
    parser.add_argument("-s", "--singlet_phase", type = float, default = 0.04, help = "The singlet semiclassical phase modulo pi in multiples of pi.")
    parser.add_argument("-t", "--triplet_phase", type = float, default = 0.24, help = "The triplet semiclassical phase modulo pi in multiples of pi.")
    args = parser.parse_args()

    number_of_parameters = 24
    all_phases = np.linspace(0.00, 1.00, (number_of_parameters+2) )[1:-1]
    SINGLETSCALING = [parameter_from_semiclassical_phase(phase, singlet_scaling_path, starting_points=[1.000,1.010]) for phase in all_phases]
    TRIPLETSCALING = [parameter_from_semiclassical_phase(phase, triplet_scaling_path, starting_points=[1.000,0.996]) for phase in all_phases]
    # scaling_combinations = itertools.product(SINGLETSCALING, TRIPLETSCALING)

    molscat_input_templates = Path(__file__).parents[1].joinpath('molscat', 'input_templates', 'RbSr+_tcpld_so_first_pt_scaling').iterdir()
    phases = ((args.singlet_phase, args.triplet_phase),)
    first_point_scaling_values = (0.25, 0.5, 0.75, 1.00, 1.25, 1.5)

    ### RUN MOLSCAT ###
    output_dirs = create_and_run_parallel(molscat_input_templates, phases, first_point_scaling_values)

    ### COLLECT S-MATRIX AND PICKLE IT ####
    output_dirs = set(output_dir for output_dir in Path(__file__).parents[1].joinpath('molscat', 'outputs', 'RbSr+_tcpld_so_first_pt_scaling', f'{nenergies}_E', f'{args.singlet_phase}_{args.triplet_phase}').iterdir() if output_dir.is_dir() )
    pickle_paths = set()
    for output_dir in output_dirs:
        s_matrix_collection, duration, output_dir, pickle_path = collect_and_pickle( output_dir, SINGLETSCALING, TRIPLETSCALING )
        pickle_paths.add(pickle_path)
        print(f"The time of gathering the outputs from {output_dir} into SMatrix object and pickling SMatrix into the file: {pickle_path} was {duration:.2f} s.")

    ### LOAD S-MATRIX, CALCULATE THE EFFECTIVE PROBABILITIES AND WRITE THEM TO .TXT FILE ###
    # pickle_path = Path(__file__).parents[1].joinpath('data_produced', 'pickles', 'RbSr+_tcpld_100_E.pickle')
    # pickle_dir = Path(__file__).parents[1].joinpath('data_produced', 'pickles', 'RbSr+_tcpld_so_first_pt_scaling', '100_E', f'{args.singlet_phase}_{args.triplet_phase}')
    # first_point_scaling_values = (0.5, 1.00, 1.25, 1.5)
    # pickle_paths = (pickle_dir.joinpath(f'{first_point_scaling:.2f}.pickle') for first_point_scaling in first_point_scaling_values)
    for pickle_path in pickle_paths:
        calculate_and_save_the_peff_parallel(pickle_path, phases[0])


if __name__ == '__main__':
    main()
