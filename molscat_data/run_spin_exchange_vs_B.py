from pathlib import Path
import os
import time
from multiprocessing import Pool
import subprocess
import re
import argparse

import numpy as np
from sigfig import round

import itertools

from matplotlib import pyplot as plt
import matplotlib as mpl

from _molscat_data.smatrix import SMatrixCollection
from _molscat_data.smatrix_compatibility_copy import SMatrixCollection as SMatrixCollectionV0
from _molscat_data.thermal_averaging import n_root_scale, n_root_distribution, n_root_iterator
from _molscat_data.utils import rate_fmfsms_vs_L_SE, rate_fmfsms_vs_L_multiprocessing, rate_fmfsms_vs_L
from _molscat_data.scaling_old import parameter_from_semiclassical_phase, default_singlet_phase_function, default_triplet_phase_function, default_singlet_parameter_from_phase, default_triplet_parameter_from_phase
from _molscat_data.analytical import MonoAlkaliEnergy
from _molscat_data.utils import probability
from _molscat_data.physical_constants import amu_to_au

from _molscat_data.visualize import PartialRateVsEnergy, RateVsMagneticField


E_min, E_max, nenergies, n = 4e-7, 4e-3, 3, 3
energy_tuple = tuple( round(n_root_scale(i, E_min, E_max, nenergies-1, n = n), sigfigs = 11) for i in range(nenergies) )
molscat_energy_array_str = str(energy_tuple).strip(')').strip('(')
scratch_path = Path(os.path.expandvars('$SCRATCH'))
pickle_dir_path = scratch_path / 'python' / 'molscat_data' / 'data_produced' / 'pickles'
pickle_dir_path.mkdir(parents=True, exist_ok=True)
arrays_dir_path = pickle_dir_path.parent / 'arrays'
arrays_dir_path.mkdir(parents=True, exist_ok=True)
plots_dir_path = scratch_path / 'python' / 'molscat_data' / 'plots'


def create_and_run_SE_vs_B(molscat_input_template_path: Path | str, singlet_phase: float, triplet_phase: float, magnetic_field: float) -> tuple[float, float, float]:
    
    time_0 = time.perf_counter()
    F1, F2 = 2, 1
    MF1, MF2 = -2, 1
    singlet_scaling = default_singlet_parameter_from_phase(singlet_phase)
    triplet_scaling = default_triplet_parameter_from_phase(triplet_phase)

    molscat_executable_path = Path.home().joinpath('molscat-RKHS', 'molscat-exe', 'molscat-alk_alk-RKHS')
    molscat_input_templates_dir_path = Path(__file__).parents[1].joinpath('molscat', 'input_templates')
    molscat_input_path = Path(__file__).parents[1].joinpath('molscat', 'inputs', molscat_input_template_path.parent.relative_to(molscat_input_templates_dir_path), f'{nenergies}_E', f'{singlet_phase:.4f}_{triplet_phase:.4f}', f'{magnetic_field}', molscat_input_template_path.stem).with_suffix('.input')
    molscat_output_path  = scratch_path.joinpath('molscat', 'outputs', molscat_input_template_path.parent.relative_to(molscat_input_templates_dir_path), f'{nenergies}_E', f'{singlet_phase:.4f}_{triplet_phase:.4f}', f'{magnetic_field}', molscat_input_template_path.stem).with_suffix('.output')
    molscat_input_path.parent.mkdir(parents = True, exist_ok = True)
    molscat_output_path.parent.mkdir(parents = True, exist_ok = True)
    
    singlet_potential_path = Path(__file__).parents[1] / 'molscat' / 'potentials' / 'singlet.dat'
    triplet_potential_path = Path(__file__).parents[1] / 'molscat' / 'potentials' / 'triplet.dat'

    with open(molscat_input_template_path, 'r') as molscat_template:
        input_content = molscat_template.read()
        input_content = re.sub("NENERGIES", str(int(nenergies)), input_content, flags = re.M)
        input_content = re.sub("ENERGYARRAY", molscat_energy_array_str, input_content, flags = re.M)
        input_content = re.sub("MFTOT", str(MF1+MF2), input_content, flags = re.M)
        input_content = re.sub("FFRb", str(F1), input_content, flags = re.M)
        input_content = re.sub("MFRb", str(MF1), input_content, flags = re.M)
        input_content = re.sub("FFSr", str(F2), input_content, flags = re.M)
        input_content = re.sub("MFSr", str(MF2), input_content, flags = re.M)
        input_content = re.sub("MAGNETICFIELD", str(magnetic_field), input_content, flags = re.M)
        input_content = re.sub("SINGLETPATH", f'\"{singlet_potential_path}\"', input_content, flags = re.M)
        input_content = re.sub("TRIPLETPATH", f'\"{triplet_potential_path}\"', input_content, flags = re.M)
        input_content = re.sub("SINGLETSCALING", str(singlet_scaling), input_content, flags = re.M)
        input_content = re.sub("TRIPLETSCALING", str(triplet_scaling), input_content, flags = re.M)

        with open(molscat_input_path, 'w') as molscat_input:
            molscat_input.write(input_content)
            molscat_input.truncate()

    molscat_command = f"{molscat_executable_path} < {molscat_input_path} > {molscat_output_path}"
    print(f"{molscat_input_path.name} for {singlet_phase=}, {triplet_phase=} run.")
    subprocess.run(molscat_command, shell = True)

    duration = time.perf_counter()-time_0
    
    return duration, molscat_input_path, molscat_output_path

def create_and_run_parallel_SE_vs_B(molscat_input_templates: tuple[str, ...], phases: tuple[tuple[float, float], ...], magnetic_fields: tuple[float, ...]) -> list[Path]:
    t0 = time.perf_counter()
    output_dirs = []
    with Pool() as pool:
       arguments = ( (x, *y, z) for x, y, z in itertools.product( molscat_input_templates, phases, magnetic_fields))
       results = pool.starmap(create_and_run_SE_vs_B, arguments)
    
       for duration, input_path, output_path in results:
           if output_path.parent not in output_dirs: output_dirs.append( output_path.parent )
           print(f"It took {duration:.2f} s to create the molscat input: {input_path}, run molscat and generate the output: {output_path}.")
    t1 = time.perf_counter()
    print(f"The time of the calculations in molscat was {t1 - t0:.2f} s.")

    return output_dirs

def collect_and_pickle_SE(molscat_output_directory_path: Path | str ) -> tuple[SMatrixCollection, float, Path, Path]:

    time_0 = time.perf_counter()
    molscat_out_dir = scratch_path.joinpath('molscat', 'outputs')
    s_matrix_collection = SMatrixCollection(collisionEnergy = energy_tuple)
    
    for output_path in Path(molscat_output_directory_path).iterdir():
        s_matrix_collection.update_from_output(file_path = output_path)


    pickle_path = pickle_dir_path / molscat_output_directory_path.relative_to(molscat_out_dir)
    pickle_path = pickle_path.parent / (pickle_path.name + '.pickle')
    pickle_path.parent.mkdir(parents = True, exist_ok = True)

    s_matrix_collection.toPickle(pickle_path)

    duration = time.perf_counter()-time_0

    return s_matrix_collection, duration, molscat_output_directory_path, pickle_path


def save_and_plot_average_vs_B(pickle_paths: tuple[Path, ...]):
    # pickle_paths = tuple(Path(pickle_dir).iterdir())
    s_matrix_collections = tuple(SMatrixCollection.fromPickle(pickle_path) for pickle_path in pickle_paths)
    phases = tuple( (default_singlet_phase_function(s_matrix_collection.singletParameter[0]), default_triplet_phase_function(s_matrix_collection.tripletParameter[0]) ) for s_matrix_collection in s_matrix_collections )
    magnetic_fields = tuple( s_matrix_collection.magneticField[0] for s_matrix_collection in s_matrix_collections )
    
    with Pool() as pool:
        arguments = ( (s_matrix_collection, 2, 0, 1, -1, 2, -2, 1, 1, None, 'cm**3/s') for s_matrix_collection in s_matrix_collections )
        k_L_E_arrays = pool.starmap(rate_fmfsms_vs_L_SE, arguments)
    
    array_paths = []
    averaged_rates = []
    for k_L_E_array, phase, pickle_path, s_matrix_collection, magnetic_field in zip(k_L_E_arrays, phases, pickle_paths, s_matrix_collections, magnetic_fields):
        array_path = arrays_dir_path / '2-211_SE_vs_E' / pickle_path.relative_to(pickle_dir_path).with_suffix('.txt')
        array_path.parent.mkdir(parents=True, exist_ok=True)
        name = f"|f = 1, m_f = -1, m_s = 1/2> to |f = 1, m_f = 0, m_s = -1/2> collisions."
        np.savetxt(array_path, k_L_E_array.reshape(k_L_E_array.shape[0], -1), fmt = '%#.10g', header = f'[Original shape: {k_L_E_array.shape}]\nThe bare (output-state-resolved) probabilities of the {name}.\nThe values of reduced mass: {np.array(s_matrix_collection.reducedMass)/amu_to_au} a.m.u.\nThe singlet, triplet semiclassical phases: ({phase[0]}, {phase[1]}). The magnetic field: {magnetic_field}.')
        array_paths.append(array_path)

        k_L_E_array = k_L_E_array.squeeze()
        total_k_E_array = k_L_E_array.sum(axis = 0)
        averaged_rates.append(s_matrix_collection.thermalAverage(total_k_E_array))
    
    fig, ax  = RateVsMagneticField.plotRate(magnetic_fields, averaged_rates)
    ax.set_title('The rate of the spin-exchange for the $\left|1,-1\\right>\hspace{0.2}\left|\\hspace{-.2}\\uparrow\\hspace{-.2}\\right>$ initial state.')
    ax.set_ylabel('rate ($\\mathrm{cm}^3/\mathrm{s})')
    ax.set_xlabel('magnetic field (G)')

    image_path = plots_dir_path / 'spin_exchange_vs_B' / f'{phase[0]:.4f}_{phase[1]:.4f}.png'
    image_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(image_path)

    return array_paths

def main():
    parser_description = "This is a python script for running molscat, collecting and pickling S-matrices, and calculating effective probabilities."
    parser = argparse.ArgumentParser(description=parser_description)
    parser.add_argument("-s", "--singlet_phase", type = float, default = 0.04, help = "The singlet semiclassical phase modulo pi in multiples of pi.")
    parser.add_argument("-t", "--triplet_phase", type = float, default = 0.24, help = "The triplet semiclassical phase modulo pi in multiples of pi.")
    parser.add_argument("--Bmin", type = float, default = 0.0)
    parser.add_argument("--Bmax", type = float, default = 98.0)
    parser.add_argument("--dB", type = float, default = 1.0)
    args = parser.parse_args()
    

    phases = ((args.singlet_phase, args.triplet_phase),)
    magnetic_fields = np.arange(args.Bmin, args.Bmax+0.1*args.dB, args.dB)

    molscat_input_templates = Path(__file__).parents[1].joinpath('molscat', 'input_templates', 'RbSr+_fmf_SE_vs_B').iterdir()

    ### RUN MOLSCAT ###
    output_dirs = create_and_run_parallel_SE_vs_B(molscat_input_templates, phases, magnetic_fields)

    ### COLLECT S-MATRIX AND PICKLE IT ####
    # output_dir = Path(__file__).parents[1].joinpath('molscat', 'outputs', 'RbSr+_tcpld_so_scaling', f'{nenergies}_E', f'{args.singlet_phase:.4f}_{args.triplet_phase:.4f}')
    # pickle_paths = [ pickle_dir_path.joinpath('RbSr+_tcpld_SE', '200_E', f'{phase[0]:.4f}_{phase[1]:.4f}.pickle') for phase in phases ]
    pickle_paths = []
    for output_dir in output_dirs:
       _, duration, output_dir, pickle_path = collect_and_pickle_SE( output_dir )
       pickle_paths.append(pickle_path)
       print(f"The time of gathering the outputs from {output_dir} into SMatrix object and pickling SMatrix into the file: {pickle_path} was {duration:.2f} s.")

    array_paths = save_and_plot_average_vs_B(pickle_paths)

if __name__ == "__main__":
    main()