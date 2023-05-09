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

from _molscat_data.smatrix import SMatrixCollection
from _molscat_data.thermal_averaging import n_root_scale, n_root_distribution, n_root_iterator
from _molscat_data.utils import rate_fmfsms_vs_L_SE, rate_fmfsms_vs_L_multiprocessing
from _molscat_data.scaling_old import parameter_from_semiclassical_phase, default_singlet_phase_function, default_triplet_phase_function, default_singlet_parameter_from_phase, default_triplet_parameter_from_phase
from _molscat_data.utils import probability

from _molscat_data.visualize import PartialRateVsEnergy

E_min, E_max, nenergies, n = 4e-7, 8e-3, 200, 2
energy_tuple = tuple( round(n_root_scale(i, E_min, E_max, nenergies-1, n = n), sigfigs = 11) for i in range(nenergies) )
molscat_energy_array_str = str(energy_tuple).strip(')').strip('(')
scratch_path = Path(os.path.expandvars('$SCRATCH'))
pickle_dir_path = scratch_path / 'python' / 'molscat_data' / 'data_produced' / 'pickles'
pickle_dir_path.mkdir(parents=True, exist_ok=True)
arrays_dir_path = pickle_dir_path.parent / 'arrays'
arrays_dir_path.mkdir(parents=True, exist_ok=True)
plots_dir_path = scratch_path / 'python' / 'molscat_data' / 'plots'


def create_and_run_SE(molscat_input_template_path: Path | str, singlet_phase: float, triplet_phase: float) -> tuple[float, float, float]:
    
    time_0 = time.perf_counter()

    singlet_scaling = default_singlet_parameter_from_phase(singlet_phase)
    triplet_scaling = default_triplet_parameter_from_phase(triplet_phase)

    molscat_executable_path = Path.home().joinpath('molscat-RKHS-tcpld', 'molscat-exe', 'molscat-RKHS-tcpld')
    molscat_input_templates_dir_path = Path(__file__).parents[1].joinpath('molscat', 'input_templates')
    molscat_input_path = Path(__file__).parents[1].joinpath('molscat', 'inputs', molscat_input_template_path.parent.relative_to(molscat_input_templates_dir_path), f'{nenergies}_E', f'{singlet_phase:.4f}_{triplet_phase:.4f}', molscat_input_template_path.stem).with_suffix('.input')
    molscat_output_path  = scratch_path.joinpath('molscat', 'outputs', molscat_input_template_path.parent.relative_to(molscat_input_templates_dir_path), f'{nenergies}_E', f'{singlet_phase:.4f}_{triplet_phase:.4f}', molscat_input_template_path.stem).with_suffix('.output')
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
    print(f"{molscat_input_path.name} for {singlet_phase=}, {triplet_phase=} run.")
    subprocess.run(molscat_command, shell = True)

    duration = time.perf_counter()-time_0
    
    return duration, molscat_input_path, molscat_output_path

def create_and_run_parallel_SE(molscat_input_templates: tuple[str, ...], phases: tuple[tuple[float, float], ...]) -> list[Path]:
    t0 = time.perf_counter()
    output_dirs = []
    with Pool() as pool:
       arguments = ( (x, *y) for x, y in itertools.product( molscat_input_templates, phases))
       results = pool.starmap(create_and_run_SE, arguments)
    
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

def save_and_plot_k_L_E_multiprocessing(pickle_paths: tuple[Path, ...]):
    with Pool() as pool:
        s_matrix_collections = pool.map(SMatrixCollection.fromPickle, pickle_paths)
        phases = tuple( (default_singlet_phase_function(s_matrix_collection.singletParameter[0]), default_triplet_phase_function(s_matrix_collection.tripletParameter[0]) ) for s_matrix_collection in s_matrix_collections )
        arguments = ( (s_matrix_collection, 2, 0, 1, -1, 2, -2, 1, 1, 'cm**3/s', None) for s_matrix_collection in s_matrix_collections )
        k_L_E_arrays = pool.starmap(rate_fmfsms_vs_L_SE, arguments)
    
    array_paths = []
    averaged_rates = []
    for k_L_E_array, phase, pickle_path, s_matrix_collection in zip(k_L_E_arrays, phases, pickle_paths, s_matrix_collections):
        array_path = arrays_dir_path / 'k_L_E' / pickle_path.relative_to(Path(__file__).parents[1] / 'data_produced' / 'pickles').with_suffix('.txt')
        array_path.parent.mkdir(parents=True, exist_ok=True)
        array_paths.append(array_path)
        np.savetxt(array_path, k_L_E_array)

        energy_array = np.array(s_matrix_collection.collisionEnergy, dtype = float)
        total_k_L_E_array = k_L_E_array.sum(axis = 0)
        averaged_rates.append(s_matrix_collection.thermalAverage(total_k_L_E_array))

        fig, ax = PartialRateVsEnergy.plotRate(energy_array, k_L_E_array)
        ax.plot(energy_array, total_k_L_E_array, label = "total", linewidth = 3, linestyle = 'solid', color = 'k')
        ax.set_title('The rate of the spin-exchange for the $\left|1,-1\\right>\hspace{0.2}\left|\\hspace{-.2}\\uparrow\\hspace{-.2}\\right>$ initial state.')
        ax.set_ylabel('rate ($\\mathrm{cm}^3/\mathrm{s})')
        ax.set_xlabel('collision energy (K)')

        image_path = plots_dir_path / 'resonance_for_MF' / f'loglog_{phase[0]:.4f}_{phase[1]:.4f}.png'
        image_path.parent.mkdir(parents=True, exist_ok=True)
        ax.set_ylim(np.max(total_k_L_E_array)*1e-3, np.max(total_k_L_E_array)*5)
        fig.savefig(image_path)

        image_path = image_path.parent / f'loglin_{phase[0]:.4f}_{phase[1]:.4f}.png'
        ax.set_yscale('linear')
        ax.set_ylim(0, np.max(total_k_L_E_array)*1.25)
        fig.savefig(image_path)

        plt.close()

    averaged_rates_path = arrays_dir_path / 'averaged_rates_vs_sum_of_phases' / f'{(phases[0][1]-phases[0][0]) % 1:.4f}.txt'
    averaged_rates = np.array([[phase[0]+phase[1] for phase in phases], averaged_rates])
    np.savetxt(averaged_rates_path, averaged_rates, fmt='%.8e', header = f'The difference of phases (triplet phase - singlet phase): {(phases[0][1]-phases[0][0]) % 1:.4f}.')

    return array_paths, averaged_rates

def plot_rate_vs_phase_sum(phase_sum, rate):
    fig, ax = plt.subplots()
    ax.plot(phase_sum, rate)
    ax.grid()
    ax.set_xlabel('$\Phi_\mathrm{s}+\Phi_\mathrm{t}$')
    ax.set_ylabel('rate ($\\mathrm{cm}^3/\mathrm{s})')
    ax.set_title('The rate of the spin-exchange for the $\left|1,-1\\right>\hspace{0.2}\left|\\hspace{-.2}\\uparrow\\hspace{-.2}\\right>$ initial state.')
    return fig, ax

def _plot_energy_grid_density(T = 5e-4, E_min = 4e-7, E_max = 4e-3, nenergies = 100, n = 3):
    fig, ax = plt.subplots()
    
    energy = np.array([ n_root_scale(i, E_min, E_max, nenergies-1, n = n) for i in range(nenergies) ])
    energy_density = 2 / (energy[2:] - energy[:-2] )
    print(energy_density.shape)
    energy_distribution = np.fromiter( n_root_iterator(temperature=T, E_min=E_min, E_max=E_max, N = nenergies, n = n), dtype = float )
    print(energy_distribution[0], energy_distribution[-1])
    ax.plot(energy[1:-1], energy_density, label = f'energy grid density ({n=})')
    ax.plot(energy, energy_distribution*np.amax(energy_density)/np.amax(energy_distribution), label = 'distribution factor')
    
    return fig, ax

def main():
    parser_description = "This is a python script for running molscat, collecting and pickling S-matrices, and calculating effective probabilities."
    parser = argparse.ArgumentParser(description=parser_description)
    parser.add_argument("-d", "--phase_difference", type = float, default = 0.00, help = "The triplet semiclassical phase minus singlet phase modulo pi.")
    parser.add_argument("--step", type = float, default = 0.01, help = "The triplet semiclassical phase minus singlet phase modulo pi.")
    args = parser.parse_args()
    
    step = args.step
    phase_difference = args.phase_difference
    triplet_phase = np.array([( phi_s + phase_difference ) % 1 for phi_s in np.arange(step, 1., step) if (phi_s + phase_difference ) % 1 != 0 ] ).round(decimals=2)
    singlet_phase = (triplet_phase - phase_difference ) % 1

    phases = np.around(tuple(zip(singlet_phase, triplet_phase, strict = True)), decimals = 2)
    molscat_input_templates = Path(__file__).parents[1].joinpath('molscat', 'input_templates', 'RbSr+_tcpld_SE').iterdir()
    
    ### RUN MOLSCAT ###
    output_dirs = create_and_run_parallel_SE(molscat_input_templates, phases)

    ### COLLECT S-MATRIX AND PICKLE IT ####
    # output_dir = Path(__file__).parents[1].joinpath('molscat', 'outputs', 'RbSr+_tcpld', f'{nenergies}_E', f'{args.singlet_phase}_{args.triplet_phase}')
    pickle_paths = []
    for output_dir in output_dirs:
        _, duration, output_dir, pickle_path = collect_and_pickle_SE( output_dir )
        pickle_paths.append(pickle_path)
        print(f"The time of gathering the outputs from {output_dir} into SMatrix object and pickling SMatrix into the file: {pickle_path} was {duration:.2f} s.")

    array_paths, averaged_rates = save_and_plot_k_L_E_multiprocessing(pickle_paths)

    fig, ax = plot_rate_vs_phase_sum(averaged_rates[0], averaged_rates[1])
    image_path = plots_dir_path / 'averaged_rates_vs_sum_of_phases' / f'{phase_difference:.4f}.txt'
    fig.savefig(image_path)
    plt.close()

    # # print(default_singlet_parameter_from_phase(.99), default_triplet_parameter_from_phase(0.99))

    # fig, ax = _plot_energy_grid_density(E_min = 4e-7, E_max=1e-2, nenergies = 200, n=2)
    # ax.set_xscale('log')
    # ax.legend()
    # plt.show()

    # singlet_scaling_path = Path(__file__).parents[1] / 'data' / 'scaling_old' / 'singlet_vs_coeff.json'
    # triplet_scaling_path = singlet_scaling_path.with_stem('triplet_vs_coeff')

    # singlet_phase, triplet_phase = 0.68, 0.84
    # phases = (singlet_phase, triplet_phase)
    # so_scaling = 1e-4

    # output_dir = Path(__file__).parents[1] / 'molscat' / 'outputs' / 'RbSr+_tcpld_so_scaling' / '100_E' / f'{singlet_phase:.4f}_{triplet_phase:.4f}' / f'{so_scaling:.4f}'
    # pickle_path = Path(__file__).parents[1] / 'data_produced' / 'pickles' / 'RbSr+_tcpld_so_scaling' / '100_E' / f'{singlet_phase:.4f}_{triplet_phase:.4f}' / f'{so_scaling:.4f}.pickle'
    # pickle_path.parent.mkdir(parents=True, exist_ok=True)
    # k_L_E_array_path = Path(__file__).parents[1] / 'data_produced' / 'arrays' / 'k_L_E' / pickle_path.relative_to(Path(__file__).parents[1] / 'data_produced' / 'pickles').with_suffix('.txt')
    # k_L_E_array_path.parent.mkdir(parents=True, exist_ok=True)
    # image_path = Path(__file__).parents[1] / 'plots' / 'resonance_for_MF' / f'{singlet_phase:.4f}_{triplet_phase:.4f}_{so_scaling:.4f}.png'
    # image_path.parent.mkdir(parents=True, exist_ok=True)

    # smatrix_collection = SMatrixCollection(spinOrbitParameter=(so_scaling,))
    # for output_path in output_dir.iterdir():
    #     smatrix_collection.update_from_output(output_path, non_molscat_so_parameter=so_scaling)

    # smatrix_collection.toPickle(pickle_path)

    # new_s_matrix_collection = SMatrixCollection.fromPickle(pickle_path)

    # # # param_indices = { "singletParameter": (new_s_matrix_collection.singletParameter.index(parameter_from_semiclassical_phase(phases[0], singlet_scaling_path, starting_points=[1.000,1.010])),), "tripletParameter": (new_s_matrix_collection.tripletParameter.index( parameter_from_semiclassical_phase(phases[1], triplet_scaling_path, starting_points=[1.000,0.996]) ), ) } if phases is not None else None

    # print("start!")
    # time_0 = time.perf_counter()

    # k_L_E_array = rate_fmfsms_vs_L_multiprocessing(new_s_matrix_collection, 2, 0, 1, -1, 2, -2, 1, 1, unit = 'cm**3/s', param_indices = None).squeeze()
    # np.savetxt(k_L_E_array_path, k_L_E_array)

    # dur = time.perf_counter() - time_0

    # print(k_L_E_array)
    # print(k_L_E_array.shape)
    # print(f'The time of calculations was {dur:.2f} s.')

    # k_L_E_array = np.loadtxt(k_L_E_array_path)
    # energy_array = np.array(new_s_matrix_collection.collisionEnergy, dtype = float)
    # total_k_L_E_array = k_L_E_array.sum(axis = 0)

    # fig, ax = PartialRateVsEnergy.plotRate(energy_array, k_L_E_array)
    # ax.plot(energy_array, total_k_L_E_array, label = "total", linewidth = 3, linestyle = 'solid', color = 'k')
    # ax.set_ylim(np.max(total_k_L_E_array)*1e-3, np.max(total_k_L_E_array)*5)
    
    # ax.set_yscale('linear')
    # ax.set_ylim(0, np.max(total_k_L_E_array)*1.25)
    
    # # ax.legend()
    # fig.savefig(image_path)
    # plt.show()

if __name__ == '__main__':
    main()