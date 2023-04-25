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

import time

from _molscat_data.smatrix import SMatrix, SMatrixCollection, CollectionParameters, CollectionParametersIndices
from _molscat_data import quantum_numbers as qn
from _molscat_data.thermal_averaging import n_root_scale
from _molscat_data.scaling_old import parameter_from_semiclassical_phase, semiclassical_phase_function, update_json
from _molscat_data.effective_probability import effective_probability
from _molscat_data.physical_constants import amu_to_au
from .prepare_so_coupling import scale_so_and_write
from .collect_V_and_SO import get_potentials_and_so
from .plot_V_and_SO import plot_so_potdiff_centrifugal

singlet_scaling_path = Path(__file__).parents[1].joinpath('data', 'scaling_old', 'singlet_vs_coeff.json')
triplet_scaling_path = Path(__file__).parents[1].joinpath('data', 'scaling_old', 'triplet_vs_coeff.json')

scratch_path = Path(os.path.expandvars('$SCRATCH'))

molscat_executable_path = Path.home().joinpath('molscat-RKHS-tcpld', 'molscat-exe', 'molscat-RKHS-tcpld')
molscat_input_templates_dir_path = Path(__file__).parents[1].joinpath('molscat', 'input_templates')
molscat_outputs_dir_path = scratch_path / 'molscat' / 'outputs'

def create_and_run(molscat_input_template_path: Path | str, singlet_phase: float = None, triplet_phase: float = None, so_scaling: float = 1.0) -> tuple[float, float, float]:
    
    molscat_input_path = Path(__file__).parents[1].joinpath('molscat', 'inputs', molscat_input_template_path.parent.relative_to(molscat_input_templates_dir_path), f'{singlet_phase:.2f}_{triplet_phase:.2f}', f'{so_scaling:.2f}', molscat_input_template_path.stem).with_suffix('.input')
    molscat_output_path  = scratch_path.joinpath('molscat', 'outputs', molscat_input_template_path.parent.relative_to(molscat_input_templates_dir_path), f'{singlet_phase:.2f}_{triplet_phase:.2f}', f'{so_scaling:.2f}', molscat_input_template_path.stem).with_suffix('.output')
    molscat_input_path.parent.mkdir(parents = True, exist_ok = True)
    molscat_output_path.parent.mkdir(parents = True, exist_ok = True)
    
    singlet_potential_path = Path(__file__).parents[1] / 'molscat' / 'potentials' / 'A.dat'
    triplet_potential_path = Path(__file__).parents[1] / 'molscat' / 'potentials' / 'a.dat'
    original_so_path = Path(__file__).parents[1] / 'data' / 'so_coupling' / 'lambda_SO_a_SrRb+_MT_original.dat'
    scaled_so_path = Path(__file__).parents[1] / 'molscat' / 'so_coupling' / molscat_input_template_path.parent.relative_to(molscat_input_templates_dir_path) / f'{singlet_phase:.2f}_{triplet_phase:.2f}' / molscat_input_template_path.stem / f'so_{so_scaling:.3f}_scaling.dat'
    scaled_so_path.parent.mkdir(parents = True, exist_ok = True)

    if singlet_phase == None:
        singlet_scaling = 1.0
    else: 
        singlet_scaling = parameter_from_semiclassical_phase(singlet_phase, singlet_scaling_path, starting_points=[1.000,1.010])
    
    if triplet_phase == None:
        triplet_scaling = 1.0
    else:
        triplet_scaling = parameter_from_semiclassical_phase(triplet_phase, triplet_scaling_path, starting_points=[1.000,0.996])

    scale_so_and_write(input_path = original_so_path, output_path = scaled_so_path, scaling = so_scaling)

    with open(molscat_input_template_path, 'r') as molscat_template:
        input_content = molscat_template.read()
        input_content = re.sub("SINGLETPATH", f'\"{singlet_potential_path}\"', input_content, flags = re.M)
        input_content = re.sub("TRIPLETPATH", f'\"{triplet_potential_path}\"', input_content, flags = re.M)
        input_content = re.sub("SOPATH", f'\"{scaled_so_path}\"', input_content, flags = re.M)
        input_content = re.sub("SINGLETSCALING", str(singlet_scaling), input_content, flags = re.M)
        input_content = re.sub("TRIPLETSCALING", str(triplet_scaling), input_content, flags = re.M)

        with open(molscat_input_path, 'w') as molscat_input:
            molscat_input.write(input_content)
            molscat_input.truncate()

    molscat_command = f"{molscat_executable_path} < {molscat_input_path} > {molscat_output_path}"
    print(f"{molscat_input_path.name} run\nwith lambda_SO scaling: {so_scaling:.3f}")
    subprocess.run(molscat_command, shell = True)

    return molscat_output_path

def main():

    molscat_input_template_path = Path(__file__).parents[1] / 'molscat' / 'input_templates' / 'plot_potentials' / 'so_scaling' / 'molscat_tcpld_RbSr+_lower.input'
    scaling_values = np.arange(0, 2, 0.1, dtype = float)

    for scaling_value in scaling_values:
        molscat_output_path = create_and_run(molscat_input_template_path = molscat_input_template_path, so_scaling = scaling_value)
        json_path = Path(__file__).parents[1] / 'data_produced' / 'json_files' / molscat_output_path.relative_to(molscat_outputs_dir_path).with_suffix('.json')
        image_path = Path(__file__).parents[1] / 'plots' / 'so_comparison'
        
        potentials_data = get_potentials_and_so(molscat_output_path)
        update_json(potentials_data, json_path = json_path)
        
        plot_so_potdiff_centrifugal(json_path, impath = image_path, L = 2)

if __name__ == '__main__':
    main()
