import subprocess
# import os
from pathlib import Path, PurePath
import re

import itertools

import numpy as np
from sigfig import round

import time

from _molscat_data.smatrix import SMatrix, SMatrixCollection, CollectionParameters, CollectionParametersIndices
from _molscat_data import quantum_numbers as qn
from _molscat_data.thermal_averaging import n_root_scale
from _molscat_data.scaling_old import parameter_from_semiclassical_phase

singlet_scaling_path = Path(__file__).parents[1].joinpath('data', 'scaling_old', 'singlet_vs_coeff.json')
triplet_scaling_path = Path(__file__).parents[1].joinpath('data', 'scaling_old', 'triplet_vs_coeff.json')
molscat_executable_path = Path.home().joinpath('molscat-RKHS-tcpld', 'molscat-exe', 'molscat-RKHS-tcpld')

E_min, E_max, nenergies, n = 4e-7, 4e-3, 10, 3
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
    
    molscat_input_path = Path(__file__).parents[1].joinpath('molscat', 'inputs', 'RbSr+_tcpld', molscat_input_template_path.name)
    molscat_output_path  = Path(__file__).parents[1].joinpath('molscat', 'outputs', 'RbSr+_tcpld', molscat_input_template_path.name).with_suffix('.output')
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

    return duration, molscat_output_directory_path, pickle_path

def main():
    from multiprocessing import Pool
    # from multiprocessing.dummy import Pool as ThreadPool 

    molscat_input_templates = Path(__file__).parents[1].joinpath('molscat', 'input_templates', 'RbSr+_tcpld').iterdir()

    time_0 = time.perf_counter()

    with Pool() as pool:
        results = pool.imap(create_and_run, molscat_input_templates)

        for duration, input_path, output_path in results:
            output_dir = output_path.parent
            print(f"It took {duration:.2f} s to create the molscat input: {input_path}, run molscat and generate the output: {output_path}.")

    duration, output_dir, pickle_path = collect_and_pickle( output_dir )
    print(f"The time of gathering the outputs from {output_dir} into SMatrix object and pickling SMatrix into the file: {pickle_path} was {duration:.2f} s.")

    total_duration = time.perf_counter()-time_0
    print(f"The total time was {total_duration:.2f} s.")
    
    print(Path.home())

if __name__ == '__main__':
    main()
