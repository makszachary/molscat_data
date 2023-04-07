import subprocess
# import os
from pathlib import Path, PurePath
import re

import numpy as np
from sigfig import round

from _molscat_data.smatrix import SMatrix, SMatrixCollection, CollectionParameters, CollectionParametersIndices
from _molscat_data import quantum_numbers as qn
from _molscat_data.thermal_averaging import n_root_scale
from _molscat_data.scaling_old import parameter_from_semiclassical_phase

def main():
    
    singlet_scaling_path = str(Path(__file__).parents[1].joinpath('data', 'scaling_old', 'singlet_vs_coeff.json'))
    triplet_scaling_path = str(Path(__file__).parents[1].joinpath('data', 'scaling_old', 'triplet_vs_coeff.json'))
    
    molscat_executable_path = Path('$HOME','molscat-RKHS-tcpld', 'molscat-exe', 'molscat-RKHS-tcpld')
    molscat_input_template_path = Path(__file__).parents[1].joinpath('molscat', 'input_templates', 'molscat-RbSr+.input')
    molscat_input_path = Path(__file__).parents[1].joinpath('molscat', 'inputs', 'molscat-RbSr+.input')
    molscat_output_path  = Path(__file__).parents[1].joinpath('molscat', 'outputs', 'molscat-RbSr+.output')
    molscat_input_path.parent.mkdir(parents = True, exist_ok = True)
    molscat_output_path.parent.mkdir(parents = True, exist_ok = True)
    print(f"""{molscat_executable_path}\n{molscat_input_template_path}""")
    E_min, E_max, nenergies, n = 4e-7, 4e-3, 10, 3

    energyarray_str = str([round(n_root_scale(i, E_min, E_max, nenergies-1, n = n), sigfigs = 11) for i in range(nenergies)]).strip(']').strip('[')
    print(energyarray_str)

    # phases = np.linspace(0.00, 1.00, (args.number_of_parameters+2) )[1:-1]
    # SINGLETSCALING = [parameter_from_semiclassical_phase(phase, Path('$HOME', 'python', 'molscat_data', 'data', 'scaling_old', 'singlet_vs_coeff.json'), starting_points=[1.000,1.010]) for phase in phases]
    # TRIPLETSCALING = [parameter_from_semiclassical_phase(phase, Path('$HOME', 'python', 'molscat_data', 'data', 'scaling_old', 'triplet_vs_coeff.json'), starting_points=[1.000,0.996]) for phase in phases]
    # combinations = [(c_s, c_t) for c_s in SINGLETSCALING for c_t in TRIPLETSCALING]
    
    singlet_phase = 0.04
    triplet_phase = 0.24
    
    singlet_scaling = parameter_from_semiclassical_phase(singlet_phase, singlet_scaling_path, starting_points=[1.000,1.010])
    lambda_t = parameter_from_semiclassical_phase(triplet_phase, triplet_scaling_path, starting_points=[1.000,0.996])

    with open(molscat_input_template_path, 'r') as molscat_template:
        input_content = molscat_template.read()
        input_content = re.sub("NENERGIES", str(int(nenergies)), input_content, flags = re.M)
        input_content = re.sub("ENERGYARRAY", energyarray_str, input_content, flags = re.M)
        input_content = re.sub("SINGLETSCALING", str(singlet_scaling), input_content, flags = re.M)
        input_content = re.sub("TRIPLETSCALING", str(lambda_t), input_content, flags = re.M)

        with open(molscat_input_path, 'w') as molscat_input:
            molscat_input.write(input_content)
            molscat_input.truncate()

    molscat_command = f"{molscat_executable_path} < {molscat_input_path} > {molscat_output_path}"
    print(molscat_command)
    subprocess.run(molscat_command, shell = True)
    print("Done!") 

if __name__ == '__main__':
    main()
