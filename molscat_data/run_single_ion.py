from typing import Any
import subprocess
from pathlib import Path, PurePath
import re

from multiprocessing import Pool
# from multiprocessing.dummy import Pool as ThreadPool 

import itertools

import numpy as np
from sigfig import round

import time

from _molscat_data.smatrix import SMatrix, SMatrixCollection, CollectionParameters, CollectionParametersIndices
from _molscat_data import quantum_numbers as qn
from _molscat_data.thermal_averaging import n_root_scale
from _molscat_data.scaling_old import parameter_from_semiclassical_phase
from _molscat_data.effective_probability import effective_probability

singlet_scaling_path = Path(__file__).parents[1].joinpath('data', 'scaling_old', 'singlet_vs_coeff.json')
triplet_scaling_path = Path(__file__).parents[1].joinpath('data', 'scaling_old', 'triplet_vs_coeff.json')
molscat_executable_path = Path.home().joinpath('molscat-RKHS-tcpld', 'molscat-exe', 'molscat-RKHS-tcpld')

E_min, E_max, nenergies, n = 4e-7, 4e-3, 100, 3
energy_tuple = tuple( round(n_root_scale(i, E_min, E_max, nenergies-1, n = n), sigfigs = 11) for i in range(nenergies) )
molscat_energy_array_str = str(energy_tuple).strip(')').strip('(')

number_of_parameters = 24

phases = np.linspace(0.00, 1.00, (number_of_parameters+2) )[1:-1]
SINGLETSCALING = [parameter_from_semiclassical_phase(phase, singlet_scaling_path, starting_points=[1.000,1.010]) for phase in phases]
TRIPLETSCALING = [parameter_from_semiclassical_phase(phase, triplet_scaling_path, starting_points=[1.000,0.996]) for phase in phases]
scaling_combinations = itertools.product(SINGLETSCALING, TRIPLETSCALING)
# [(c_s, c_t) for c_s in SINGLETSCALING for c_t in TRIPLETSCALING]

def create_and_run(molscat_input_template_path):
    
    time_0 = time.perf_counter()
    
    molscat_input_path = Path(__file__).parents[1].joinpath('molscat', 'inputs', 'RbSr+_tcpld_100_E', molscat_input_template_path.name)
    molscat_output_path  = Path(__file__).parents[1].joinpath('molscat', 'outputs', 'RbSr+_tcpld_100_E', molscat_input_template_path.name).with_suffix('.output')
    molscat_input_path.parent.mkdir(parents = True, exist_ok = True)
    molscat_output_path.parent.mkdir(parents = True, exist_ok = True)
    
    singlet_phase = 0.04
    triplet_phase = 0.24
    
    singlet_scaling = parameter_from_semiclassical_phase(singlet_phase, singlet_scaling_path, starting_points=[1.000,1.010])
    # singlet_scaling = round(singlet_scaling, sigfigs = 12)
    triplet_scaling = parameter_from_semiclassical_phase(triplet_phase, triplet_scaling_path, starting_points=[1.000,0.996])
    # triplet_scaling = round(triplet_scaling, sigfigs = 12)

    with open(molscat_input_template_path, 'r') as molscat_template:
        input_content = molscat_template.read()
        input_content = re.sub("NENERGIES", str(int(nenergies)), input_content, flags = re.M)
        input_content = re.sub("ENERGYARRAY", molscat_energy_array_str, input_content, flags = re.M)
        input_content = re.sub("SINGLETSCALING", str(singlet_scaling), input_content, flags = re.M)
        input_content = re.sub("TRIPLETSCALING", str(triplet_scaling), input_content, flags = re.M)

        with open(molscat_input_path, 'w') as molscat_input:
            molscat_input.write(input_content)
            molscat_input.truncate()

    molscat_command = f"{molscat_executable_path} < {molscat_input_path} > {molscat_output_path}"
    # print(molscat_command)
    print(f"{molscat_input_path.name} run")
    subprocess.run(molscat_command, shell = True)
    # print("Molscat done!")

    duration = time.perf_counter()-time_0
    
    return duration, molscat_input_template_path, molscat_output_path

def collect_and_pickle(molscat_output_directory_path):

    time_0 = time.perf_counter()

    s_matrix_collection = SMatrixCollection(singletParameter = SINGLETSCALING, tripletParameter = TRIPLETSCALING, collisionEnergy = energy_tuple)
    
    for output_path in Path(molscat_output_directory_path).iterdir():
        s_matrix_collection.update_from_output(file_path = output_path)
    
    pickle_path = Path(__file__).parents[1].joinpath('data_produced', molscat_output_directory_path.with_suffix('.pickle').name)
    pickle_path.parent.mkdir(parents = True, exist_ok = True)

    s_matrix_collection.toPickle(pickle_path)

    duration = time.perf_counter()-time_0

    return duration, s_matrix_collection, molscat_output_directory_path, pickle_path

def rate_fmfms(s_matrix_collection: SMatrixCollection, F_out: int, MF_out: int, MS_out: int, F_in: int, MF_in: int, MS_in: int, param_indices: dict) -> float:
    L_max = max(key[0].L for s_matrix in s_matrix_collection.matrixCollection.values() for key in s_matrix.matrix.keys())
    rate = np.sum( s_matrix_collection.getRateCoefficient(qn.LF1F2(L, ML, F1 = F_out, MF1 = MF_out, F2 = 1, MF2 = MS_out), qn.LF1F2(L, ML, F1 = F_in, MF1 = MF_in, F2 = 1, MF2 = MS_in), param_indices = param_indices) for L in range(0, L_max+1, 2) for ML in range(-L, L+1, 2) )
    return rate


def probability(s_matrix_collection: SMatrixCollection, F_out: int | np.ndarray[Any, int], MF_out: int | np.ndarray[Any, int], MS_out: int | np.ndarray[Any, int], F_in: int | np.ndarray[Any, int], MF_in: int | np.ndarray[Any, int], MS_in: int | np.ndarray[Any, int]) -> np.ndarray[Any, float]:
    
    args = locals().copy()
    args.pop('s_matrix_collection')
    arg_shapes = tuple( value.shape for value in args.values() if isinstance(value, np.ndarray) )

    param_indices = { "singletParameter": (0,), "tripletParameter": (5,) }

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
           results = pool.starmap(rate_fmfms, arguments)
           rate_shape = results[0].shape
           rate = np.array(results).reshape((*arg_shapes[0], *rate_shape))

           averaged_rate = s_matrix_collection.thermalAverage(rate)
           averaged_momentum_transfer_rate = np.full_like(averaged_rate, averaged_momentum_transfer_rate)
           probability = averaged_rate / averaged_momentum_transfer_rate

           return probability
    
    rate = rate_fmfms(s_matrix_collection, **args)
    averaged_rate = s_matrix_collection.thermalAverage(rate)
    probability = averaged_rate / averaged_momentum_transfer_rate

    return probability

        
    
    # results = rate_fmfms()
        # results = np.fromiter( ( rate_fmfms(s_matrix_collection, *(args[name][index] for name in args)) for index in np.ndindex(arg_shapes[0]) ), dtype= float)
        # print(results)
    # print('fuck you')
        

    # args = locals().copy()
    # args.pop('s_matrix_collection')
    # print(dict(args))    
    # print(arg_shapes)
    # print(all(qn_shape == arg_shapes[0] for qn_shape in arg_shapes))
    
    # param_indices = { "singletParameter": (0,), "tripletParameter": (5,) }
    
    # L_max = max(key[0].L for s_matrix in s_matrix_collection.matrixCollection.values() for key in s_matrix.matrix.keys())
    
    # arguments = ( ( qn_number if isinstance(value, np.ndarray) else value for value in args.values()) for qn_number in value.flatten() )
    # print(arguments)
    # print(tuple(arguments))

    # with Pool() as pool:
    #    results = pool.starmap(rate_fmfms, args)

    #    for name, result in zip(names, results):
    #        print(result)

    # rate = sum( s_matrix_collection.getRateCoefficient(qn.LF1F2(L, ML, F1 = F_out, MF1 = MF_out, F2 = 1, MF2 = MS_out), qn.LF1F2(L, ML, F1 = F_in, MF1 = MF_in, F2 = 1, MF2 = MS_in), param_indices = param_indices) for L in range(0, L_max+1, 2) for ML in range(-L, L+1, 2) )
    # averaged_rate = s_matrix_collection.thermalAverage(rate)
    
    # averaged_momentum_transfer_rate = s_matrix_collection.getThermallyAveragedMomentumTransferRate(qn.LF1F2(None, None, F1 = 2, MF1 = 2, F2 = 1, MF2 = -1), param_indices = param_indices)
    
    # probability = averaged_rate/averaged_momentum_transfer_rate

    # duration = time.perf_counter()-time_0
    # print(f'{duration} s.')

    # return probability


def main():

    molscat_input_templates = Path(__file__).parents[1].joinpath('molscat', 'input_templates', 'RbSr+_tcpld').iterdir()
    pickle_path = Path(__file__).parents[1].joinpath('data_produced', 'RbSr+_tcpld_100_E.pickle')
    # pickle_path = Path(__file__).parents[1].joinpath('data_produced', 'RbSr+_tcpld.pickle')
    # pickle_path = Path(__file__).parents[1].joinpath('data_produced', 'json_test_3.pickle')

    time_0 = time.perf_counter()

    #with Pool() as pool:
    #    results = pool.imap(create_and_run, molscat_input_templates)
    #
    #    for duration, input_path, output_path in results:
    #        output_dir = output_path.parent
    #        print(f"It took {duration:.2f} s to create the molscat input: {input_path}, run molscat and generate the output: {output_path}.")

    #print(f"The time of the calculations in molscat was {time.perf_counter() - time_0:.2f} s.")

    #duration, s_matrix_collection, output_dir, pickle_path = collect_and_pickle( output_dir )
    #print(f"The time of gathering the outputs from {output_dir} into SMatrix object and pickling SMatrix into the file: {pickle_path} was {duration:.2f} s.")

    t0 = time.perf_counter()
    s_matrix_collection = SMatrixCollection.fromPickle(pickle_path)
    # print(s_matrix_collection)
    
    pmf_path = Path(__file__).parents[1].joinpath('data', 'pmf', 'N_pdf_logic_params_EMM_500uK.txt')
    pmf_array = np.loadtxt(pmf_path)

    F_out, F_in, S = 2, 4, 1
    MF_out, MS_out, MF_in, MS_in = np.meshgrid(np.arange(-F_out, F_out+1, 2), np.arange(-S, S+1, 2), np.arange(-F_in, F_in+1, 2), S, indexing = 'ij')
    arg_hpf_deexcitation = (s_matrix_collection, F_out, MF_out, MS_out, F_in, MF_in, MS_in)

    F_out, F_in, S = 4, 4, 1
    MF_out, MS_out, MF_in, MS_in = np.meshgrid(np.arange(-F_out, F_out+1, 2), -S, np.arange(-F_in, F_in+1, 2), S, indexing = 'ij')
    arg_cold_spin_change_higher = (s_matrix_collection, F_out, MF_out, MS_out, F_in, MF_in, MS_in)

    F_out, F_in, S = 2, 2, 1
    MF_out, MS_out, MF_in, MS_in = np.meshgrid(np.arange(-F_out, F_out+1, 2), -S, np.arange(-F_in, F_in+1, 2), S, indexing = 'ij')
    arg_cold_spin_change_lower = (s_matrix_collection, F_out, MF_out, MS_out, F_in, MF_in, MS_in)

    args = [arg_hpf_deexcitation, arg_cold_spin_change_higher, arg_cold_spin_change_lower]
    names = [f'hyperfine deexcitation for the |f = 2, m_f = {{-2, -1, 0, 1, 2}}> |m_s = 1/2> initial states', 
             f'cold spin change for the |f = 2, m_f = {{-2, -1, 0, 1, 2}}> |m_s = 1/2> initial states',
             f'cold spin change for the |f = 1, m_f = {{-1, 0, 1}}> |m_s = 1/2> initial states']
    abbreviations = ['hpf', 'cold_higher', 'cold_lower']

    t4 = time.perf_counter()

    for abbreviation, name, arg in zip(*map(reversed, (abbreviations, names, args) ) ) :
        print(abbreviation, name, arg)

    for abbreviation, name, arg in zip(*map(reversed, (abbreviations, names, args) ) ) :
        t = time.perf_counter()
        probability_array = probability(*arg).sum(axis = (0, 1)).squeeze()
        effective_probability_array = effective_probability(probability_array, pmf_array)

        print("------------------------------------------------------------------------")
        print(f'The bare probabilities p_0 for the {name} are:')
        print(probability_array, '\n')

        print(f'The effective probabilities p_eff for the {name} are:')
        print(effective_probability_array)
        print("------------------------------------------------------------------------")
        
        txt_path = pickle_path.parent.joinpath('arrays', pickle_path.stem+'_'+abbreviation).with_suffix('.txt')
        txt_path.parent.mkdir(parents = True, exist_ok = True)
        np.savetxt(txt_path, effective_probability_array, fmt = '%.10f')
        
        duration = time.perf_counter() - t
        print(f"It took {duration:.2f} s.")
    
    t5 = time.perf_counter()
    print(f"The time of the loop calculations with parallelized probability function was {t5 - t4:.2f} s.")


if __name__ == '__main__':
    main()
