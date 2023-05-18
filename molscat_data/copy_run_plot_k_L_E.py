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

from _molscat_data.visualize import PartialRateVsEnergy

E_min, E_max, nenergies, n = 4e-7, 4e-3, 200, 3
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

def save_and_plot_k_L_E_spinspin(pickle_path: Path | str):
    s_matrix_collection = SMatrixCollection.fromPickle(file_path=pickle_path)
    phase = (default_singlet_phase_function(s_matrix_collection.singletParameter[0]), default_triplet_phase_function(s_matrix_collection.tripletParameter[0]))
    spin_orbit_scaling = s_matrix_collection.spinOrbitParameter[0]
    k_L_E_arrays = np.array([rate_fmfsms_vs_L_multiprocessing(s_matrix_collection, 4, MF_out, 1, -1, 4, 4, 1, 1, unit = 'cm**3/s') for MF_out in range(-4, 4+1, 2) ] )
    energy_array = np.array(s_matrix_collection.collisionEnergy)
    
    with Pool() as pool:
        arguments = ((s_matrix_collection, k_L_E_array) for k_L_E_array in k_L_E_arrays)
        averaged_rates = pool.starmap(av_rate, arguments)

    for MF_out, k_L_E_array, average_rate in zip(range(-4, 4+1, 2), k_L_E_arrays, averaged_rates):
        total_k_L_E_array = k_L_E_array.sum(axis=0).squeeze() 
        k_L_E_array = k_L_E_array.squeeze()

        fig, ax = plot_k_L_E(energy_array, k_L_E_array)

        image_path = plots_dir_path / '4411_cold_vs_E' / f'MF_out_{MF_out}' / f'{(phase[1]-phase[0]) % 1:.4f}' / f'loglog_{phase[0]:.4f}_{phase[1]:.4f}_{spin_orbit_scaling:.4f}.png'
        image_path.parent.mkdir(parents=True, exist_ok=True)
        ax.set_ylim(np.min([10**(-14), np.max(total_k_L_E_array)*1e-3]), np.max([np.max(total_k_L_E_array)*5, 3*10**(-9)]))
        ax.set_yscale('log')
        ax.set_title(f'The rate of the cold ion\'s spin flip for the $\\left|2,2\\right>\hspace{{0.2}}\\left|\\hspace{{-.2}}\\uparrow\\hspace{{-.2}}\\right>$ initial state.\n$(\\tilde{{\\Phi}}_\\mathrm{{s}}, \\tilde{{\\Phi}}_\\mathrm{{t}}) = ({phase[0]:.2f}, {phase[1]:.2f}), c_\\mathrm{{so}} = {spin_orbit_scaling:.4f}$.', fontsize = 'x-large')
        ax.plot(energy_array, np.full_like(energy_array, average_rate), linewidth = 4, linestyle='--', color = 'dodgerblue', label = 'thermal average')
        for l, en, ra in get_L_label_coords(energy_array, k_L_E_array):
            ax.text(en*1., ra*1.5, f'{l}', fontsize = 'large', color = mpl.colormaps['cividis'](l/30), fontweight = 'bold', va = 'center', ha = 'center')

        fig.savefig(image_path)
        plt.close()

        fig, ax = plot_k_L_E(energy_array, k_L_E_array)

        image_path = image_path.parent / f'loglin_{phase[0]:.4f}_{phase[1]:.4f}_{spin_orbit_scaling:.4f}.png'
        ax.set_yscale('linear')
        ax.set_ylim(0, np.max([np.max(total_k_L_E_array)*1.25, 3*10**(-9)]))
        ax.set_title(f'The rate of the cold ion\'s spin flip for the $\\left|2,2\\right>\hspace{{0.2}}\\left|\\hspace{{-.2}}\\uparrow\\hspace{{-.2}}\\right>$ initial state.\n$(\\tilde{{\\Phi}}_\\mathrm{{s}}, \\tilde{{\\Phi}}_\\mathrm{{t}}) = ({phase[0]:.2f}, {phase[1]:.2f}), c_\\mathrm{{so}} = {spin_orbit_scaling:.2f}$.', fontsize = 'x-large')
        ax.plot(energy_array, np.full_like(energy_array, average_rate), linewidth = 4, linestyle='--', color = 'dodgerblue', label = 'thermal average')
        for l, en, ra in get_L_label_coords(energy_array, k_L_E_array):
            ax.text(en, ra + (ax.get_ylim()[1]-ax.get_ylim()[0])*0.02, f'{l}', fontsize = 'large', color = mpl.colormaps['cividis'](l/30), fontweight = 'bold', va = 'center', ha = 'center')
        fig.savefig(image_path)

        plt.close()

def save_and_plot_k_L_E_multiprocessing(pickle_paths: tuple[Path, ...]):
    with Pool() as pool:
        s_matrix_collections = pool.map(SMatrixCollection.fromPickle, pickle_paths)
        phases = tuple( (default_singlet_phase_function(s_matrix_collection.singletParameter[0]), default_triplet_phase_function(s_matrix_collection.tripletParameter[0]) ) for s_matrix_collection in s_matrix_collections )
        arguments = ( (s_matrix_collection, 2, 0, 1, -1, 2, -2, 1, 1, None, 'cm**3/s') for s_matrix_collection in s_matrix_collections )
        k_L_E_arrays = pool.starmap(rate_fmfsms_vs_L_SE, arguments)
    
    array_paths = []
    averaged_rates = []
    for k_L_E_array, phase, pickle_path, s_matrix_collection in zip(k_L_E_arrays, phases, pickle_paths, s_matrix_collections):
        array_path = arrays_dir_path / 'k_L_E' / pickle_path.relative_to(pickle_dir_path).with_suffix('.txt')
        array_path.parent.mkdir(parents=True, exist_ok=True)
        array_paths.append(array_path)
        k_L_E_array = k_L_E_array.squeeze()
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
        ax.set_ylim(np.min([10**(-14), np.max(total_k_L_E_array)*1e-3]), np.max([np.max(total_k_L_E_array)*5, 5*10**(-9)]))
        fig.savefig(image_path)

        image_path = image_path.parent / f'loglin_{phase[0]:.4f}_{phase[1]:.4f}.png'
        ax.set_yscale('linear')
        ax.set_ylim(0, np.max([np.max(total_k_L_E_array)*1.25, 5*10**(-9)]))
        fig.savefig(image_path)

        plt.close()

    averaged_rates_path = arrays_dir_path / 'averaged_rates_vs_sum_of_phases' / f'{(phases[0][1]-phases[0][0]) % 1:.4f}.txt'
    averaged_rates_path.parent.mkdir(parents=True, exist_ok=True)
    averaged_rates = np.array([[phase[0] for phase in phases], [phase[1] for phase in phases], averaged_rates])
    np.savetxt(averaged_rates_path, averaged_rates, fmt='%.8e', header = f'The difference of phases (triplet phase - singlet phase) is fixed: {(phases[0][1]-phases[0][0]) % 1:.4f}.')

    return array_paths, averaged_rates

def av_rate(s_matrix_collection, k_L_E_array):
    k_E_array = k_L_E_array.sum(axis=0).squeeze()
    return s_matrix_collection.thermalAverage(k_E_array)

def plot_k_L_E_vs_Phi_s(phases, k_L_E_array_dir):
    # pickle_paths = ( Path(pickle_dir) / f'{phase[0]:.4f}_{phase[1]:.4f}.pickle' for phase in phases )
    k_L_E_array_paths = ( Path(k_L_E_array_dir)/ f'{phase[0]:.4f}_{phase[1]:.4f}.txt' for phase in phases )
    energy_array = np.array([ round(n_root_scale(i, E_min, E_max, nenergies-1, n = n), sigfigs = 11) for i in range(nenergies) ])
    phi_s_array = np.array([phase[0] for phase in phases])
    k_L_E_arrays = np.array([ np.loadtxt(arr_path) for arr_path in k_L_E_array_paths ] ).transpose(1,2,0)
    phase_difference = (phases[0][1]-phases[0][0]) % 1
    # energy = 7.5e-4
    # E_index = np.abs(energy_array-energy).argmin()

    filter_max_arr = np.equal(np.full_like(k_L_E_arrays.transpose(2,0,1), np.amax(k_L_E_arrays, axis = 2)).transpose(1,2,0), k_L_E_arrays)
    

    for E_index in range(k_L_E_arrays.shape[1]):
        energy = energy_array[E_index]
        fig, ax = plt.subplots()
        ax.set_xlabel(r"$(\Phi_\mathrm{s} + \pi/4)\,\mathrm{mod}\,\pi \hspace{0.5} / \hspace{0.5} \pi$", fontsize = 'large')
        ax.set_ylabel('rate ($\\mathrm{cm}^3/\\mathrm{s}$)', fontsize = 'large')
        ax.set_xlim(0,1)
        ax.set_ylim(0, 1.2*np.amax(k_L_E_arrays[:,E_index,:]) )
        
        for L in range(k_L_E_arrays.shape[0]):
            ax.plot(phi_s_array, k_L_E_arrays[L, E_index], color = mpl.colormaps['cividis'](L/30))
        
        coords_vs_L = tuple( (l, phases[filter_max_arr[l, E_index]], k_L_E_arrays[l, E_index][filter_max_arr[l, E_index]]) for l in range(k_L_E_arrays.shape[0]) if np.any(filter_max_arr[l, E_index]) and np.any(k_L_E_arrays[l, E_index][filter_max_arr[l, E_index]] > 0.05*np.amax(k_L_E_arrays[:,E_index,:])) )
        for coord in coords_vs_L:
            ax.text(coord[1].flatten()[0], coord[2] + (ax.get_ylim()[1]-ax.get_ylim()[0])*0.02, f'{coord[0]}', fontsize = 'large', color = mpl.colormaps['cividis'](coord[0]/30), fontweight = 'bold', va = 'center', ha = 'center')

        ax.set_title(f'The $\\left|1,-1\\right>\\hspace{{0.2}}\\left|\\hspace{{-.2}}\\uparrow\\hspace{{-.2}}\\right> \\rightarrow \left|1,0\\right>\\hspace{{0.2}}\\left|\\hspace{{-.2}}\\downarrow\\hspace{{-.2}}\\right>$ collision rate.\n$(\\Phi_\\mathrm{{t}}-\\Phi_\\mathrm{{s}}) = {phase_difference:.4f} \\pi\\,\\mathrm{{mod}}\\,\\pi,\\hspace{{0.5}} E_\\mathrm{{col}} = {energy:.2e}\\,\\mathrm{{K}}$.')
        plt.tight_layout()
        image_path = Path(__file__).parents[1] / 'plots' / 'ascratch' / 'partial_rate_vs_phase' / f'{phase_difference:.4f}' / f'{energy:.8f}.png'
        image_path.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(image_path)
        plt.close()
        # plt.show()
        
        # s_matrix_collections = tuple(SMatrixCollection.fromPickle(pickle_path) for pickle_path in pickle_paths)
        # energy_arrays = tuple( np.array(s.collisionEnergy) for s in s_matrix_collections )
        
        # arguments = tuple( zip(s_matrix_collections, k_L_E_arrays) )
        
        # averaged_rates = pool.starmap(av_rate, arguments)
        # print(len(averaged_rates))
        # print("rates averaged")

def only_save_average(phases, pickle_dir, k_L_E_array_dir, energy_threshold: float = None):
    with Pool() as pool:
        pickle_paths = ( Path(pickle_dir) / f'{phase[0]:.4f}_{phase[1]:.4f}.pickle' for phase in phases )
        k_L_E_array_paths = ( Path(k_L_E_array_dir)/ f'{phase[0]:.4f}_{phase[1]:.4f}.txt' for phase in phases )
        
        k_L_E_arrays = tuple( np.loadtxt(arr_path) for arr_path in k_L_E_array_paths )
        print("arrays loaded")
        
        s_matrix_collections = tuple(SMatrixCollection.fromPickle(pickle_path) for pickle_path in pickle_paths)
        energy_arrays = tuple( np.array(s.collisionEnergy) for s in s_matrix_collections )
        
        if energy_threshold is not None:
            k_L_E_arrays = tuple( np.where(energy_array > energy_threshold, arr, 0) for arr, energy_array in zip(k_L_E_arrays, energy_arrays) )
        
        arguments = tuple( zip(s_matrix_collections, k_L_E_arrays) )
        
        averaged_rates = pool.starmap(av_rate, arguments)
        print(len(averaged_rates))
        print("rates averaged")

    for (phase, energy_array, k_L_E_array) in zip(phases, energy_arrays, k_L_E_arrays):
        total_k_L_E_array = k_L_E_array.sum(axis=0).squeeze() 

        fig, ax = plot_k_L_E(energy_array, k_L_E_array)

        image_path = plots_dir_path / 'resonance_for_MF' / f'{(phase[1]-phase[0]) % 1:.4f}' / f'loglog_{phase[0]:.4f}_{phase[1]:.4f}.png'
        image_path.parent.mkdir(parents=True, exist_ok=True)
        ax.set_ylim(np.min([10**(-14), np.max(total_k_L_E_array)*1e-3]), np.max([np.max(total_k_L_E_array)*5, 5*10**(-9)]))
        ax.set_yscale('log')
        for l, en, ra in get_L_label_coords(energy_array, k_L_E_array):
            ax.text(en*1., ra*1.5, f'{l}', fontsize = 'large', color = mpl.colormaps['cividis'](l/30), fontweight = 'bold', va = 'center', ha = 'center')

        fig.savefig(image_path)
        plt.close()

        fig, ax = plot_k_L_E(energy_array, k_L_E_array)

        image_path = image_path.parent / f'loglin_{phase[0]:.4f}_{phase[1]:.4f}.png'
        ax.set_yscale('linear')
        ax.set_ylim(0, np.max([np.max(total_k_L_E_array)*1.25, 5*10**(-9)]))
        for l, en, ra in get_L_label_coords(energy_array, k_L_E_array):
            ax.text(en, ra + (ax.get_ylim()[1]-ax.get_ylim()[0])*0.02, f'{l}', fontsize = 'large', color = mpl.colormaps['cividis'](l/30), fontweight = 'bold', va = 'center', ha = 'center')
        fig.savefig(image_path)

        plt.close()

    averaged_rates = np.array(averaged_rates).squeeze()
    print(averaged_rates.shape)
    averaged_rates_path = arrays_dir_path / 'averaged_rates_vs_sum_of_phases' / f'{(phases[0][1]-phases[0][0]) % 1:.4f}.txt'
    averaged_rates_path.parent.mkdir(parents=True, exist_ok=True)
    print(len(phases), averaged_rates.shape)
    averaged_rates = np.array([[phase[0] for phase in phases], [phase[1] for phase in phases], averaged_rates]).squeeze()
    np.savetxt(averaged_rates_path, averaged_rates, fmt='%.8e', header = f'The difference of phases (triplet phase - singlet phase) is fixed: {(phases[0][1]-phases[0][0]) % 1:.4f}.')

    return averaged_rates
        

def plot_rate_vs_singlet_phase(singlet_phase, rate):
    fig, ax = plt.subplots()
    ax.plot(singlet_phase, rate)
    ax.grid()
    ax.set_xlabel(r"$(\Phi_\mathrm{s} + \pi/4)\,\mathrm{mod}\,\pi \hspace{0.5} / \hspace{0.5} \pi$", fontsize = 'large')
    ax.set_ylabel('rate ($\\mathrm{cm}^3/\\mathrm{s}$)', fontsize = 'large')
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

def plot_k_L_E(energy_array, k_L_E_array):
    total_k_L_E_array = k_L_E_array.sum(axis=0).squeeze() 
    fig, ax = PartialRateVsEnergy.plotRate(energy_array, k_L_E_array)
    ax.plot(energy_array, total_k_L_E_array, label = "total", linewidth = 3, linestyle = 'solid', color = 'k')
    ax.set_title('The rate of the spin-exchange for the $\left|1,-1\\right>\hspace{0.2}\left|\\hspace{-.2}\\uparrow\\hspace{-.2}\\right>$ initial state.', fontsize = 'x-large')
    ax.set_ylabel('rate ($\\mathrm{cm}^3 / \\mathrm{s}$)', fontsize = 'x-large')
    ax.set_xlabel('collision energy (K)', fontsize = 'x-large')
    return fig, ax

def get_L_label_coords(energy_array, k_L_E_array):
    energy_array, k_L_E_array = np.array(energy_array), np.array(k_L_E_array)
    total_k_L_E_array = k_L_E_array.sum(axis = 0)
    filter_max_arr = np.equal(np.full_like(k_L_E_array.transpose(), np.amax(k_L_E_array, axis = 1)).transpose(), k_L_E_array)
    filter_enough_arr = k_L_E_array > 0.1*total_k_L_E_array
    filter_max_enough_arr = np.logical_and(filter_max_arr, filter_enough_arr)    
    coords_vs_l = tuple( (l, energy_array[filter_max_enough_arr[l]], k_L_E_array[l][filter_max_enough_arr[l]]) for l in range(filter_max_enough_arr.shape[0]) if np.any(filter_max_enough_arr[l]))
    
    return coords_vs_l
      

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
    # output_dir = Path(__file__).parents[1].joinpath('molscat', 'outputs', 'RbSr+_tcpld_so_scaling', f'{nenergies}_E', f'{args.singlet_phase:.4f}_{args.triplet_phase:.4f}')
    # pickle_paths = [ pickle_dir_path.joinpath('RbSr+_tcpld_SE', '200_E', f'{phase[0]:.4f}_{phase[1]:.4f}.pickle') for phase in phases ]
    # pickle_paths = []
    # for output_dir in output_dirs:
    #    _, duration, output_dir, pickle_path = collect_and_pickle_SE( output_dir )
    #    pickle_paths.append(pickle_path)
    #    print(f"The time of gathering the outputs from {output_dir} into SMatrix object and pickling SMatrix into the file: {pickle_path} was {duration:.2f} s.")

    # array_paths, averaged_rates = save_and_plot_k_L_E_multiprocessing(pickle_paths)

    phase = phases[0]
    spin_orbit_scaling = 0.38
    pickle_path = Path(__file__).parents[1] / 'data_produced' / 'pickles' / 'test_so_in_smatrix' / f'2_E' / f'{phase[0]:.4f}_{phase[1]:.4f}' / f'{spin_orbit_scaling:.4f}.pickle'
    save_and_plot_k_L_E_spinspin(pickle_path)

    ######## only plotting

    # B = 2.97
    # energy_threshold = (MonoAlkaliEnergy(B, f = 1, mf = 0, i = 3/2) + MonoAlkaliEnergy(B, f = 1/2, mf = 1/2, i = 0)) - ( MonoAlkaliEnergy(B, f = 1, mf = 1, i = 3/2) + MonoAlkaliEnergy(B, f = 1/2, mf = -1/2, i = 0) )
    # print(f'Thermal averages will be calculated for the endothermic channel with the energy threshold  = {energy_threshold:.4e} K.')

    # k_L_E_pickles_dir = pickle_dir_path / 'RbSr+_tcpld_SE' / f'{nenergies}_E'
    # k_L_E_arrays_dir = arrays_dir_path / 'k_L_E' / 'RbSr+_tcpld_SE' / f'{nenergies}_E'

    # averaged_rates = only_save_average(phases, k_L_E_pickles_dir, k_L_E_arrays_dir, energy_threshold = energy_threshold)

    # fig, ax = plot_rate_vs_singlet_phase(averaged_rates[0], averaged_rates[2])
    # ax.set_title(f'The $\\left|1,-1\\right>\\hspace{{0.2}}\\left|\\hspace{{-.2}}\\uparrow\\hspace{{-.2}}\\right> \\rightarrow \left|1,0\\right>\\hspace{{0.2}}\\left|\\hspace{{-.2}}\\downarrow\\hspace{{-.2}}\\right>$ collision rate.\n$(\\Phi_\\mathrm{{t}}-\\Phi_\\mathrm{{s}}) = {phase_difference} \\pi\\,\\mathrm{{mod}}\\,\\pi$.')
    # ax.set_xlim(0, 1)
    # ax.set_ylim(0, max(10**(-9), max(averaged_rates[2])))
    # plt.tight_layout()
    # image_path = plots_dir_path / 'averaged_rates_vs_sum_of_phases' / f'{phase_difference:.4f}.png'
    # image_path.parent.mkdir(parents=True,exist_ok=True)
    # fig.savefig(image_path)
    # plt.close()

    #######

    ####### Plot k_L,E as a function of Phi_s

    # k_L_E_array_dir = Path(__file__).parents[1] / 'data_produced' / 'arrays' / 'ascratch' / 'k_L_E' / 'RbSr+_tcpld_SE' / f'{nenergies}_E'
    # plot_k_L_E_vs_Phi_s(phases, k_L_E_array_dir)

    #######

    # energy_array = np.array([ round(n_root_scale(i, E_min, E_max, nenergies-1, n = n), sigfigs = 11) for i in range(nenergies) ])
    # array_path = Path(__file__).parents[1] / 'data_produced' / 'arrays' / 'k_L_E' / 'trash1' / '0.0900_0.2900.txt'
    # k_L_E_array = np.loadtxt(array_path)
    # k_L_E_array = k_L_E_array.squeeze()
    # total_k_L_E_array = k_L_E_array.sum(axis = 0)
    # # np.savetxt(array_path, k_L_E_array)

    # filter_max_arr = np.equal(np.full_like(k_L_E_array.transpose(), np.amax(k_L_E_array, axis = 1)).transpose(), k_L_E_array)
    # filter_enough_arr = k_L_E_array > 0.1*total_k_L_E_array
    # filter_max_enough_arr = np.logical_and(filter_max_arr, filter_enough_arr)
    # np.savetxt(array_path.parent / (array_path.with_suffix('').name + '_filter_max_enough'), filter_max_enough_arr, fmt = '%5i' )


    # # fig, ax = PartialRateVsEnergy.plotRate(energy_array, k_L_E_array)
    # # ax.plot(energy_array, total_k_L_E_array, label = "total", linewidth = 3, linestyle = 'solid', color = 'k')
    # # ax.set_title('The rate of the spin-exchange for the $\left|1,-1\\right>\hspace{0.2}\left|\\hspace{-.2}\\uparrow\\hspace{-.2}\\right>$ initial state.')
    # # ax.set_ylabel('rate ($\\mathrm{cm}^3/\mathrm{s}$)')
    # # ax.set_xlabel('collision energy (K)')

    # fig, ax = plot_k_L_E(energy_array, k_L_E_array)

    # props = dict(boxstyle = 'round', pad = .2, facecolor = 'white', edgecolor = 'k', alpha = 1.)
    # props = None

    # for l, en, ra in get_L_label_coords(energy_array, k_L_E_array):
    #     ax.text(en*1., ra*1.5, f'{l}', fontsize = 'large', color = mpl.colormaps['cividis'](l/30), fontweight = 'bold', va = 'center', ha = 'center', bbox = props)

    # image_path = plots_dir_path / 'resonance_for_MF' / f'loglog_{array_path.with_suffix("").name}.png'
    # image_path.parent.mkdir(parents=True, exist_ok=True)
    # ax.set_ylim(np.min([10**(-14), np.max(total_k_L_E_array)*1e-3]), np.max([np.max(total_k_L_E_array)*5, 5*10**(-9)]))
    # fig.savefig(image_path)
    # plt.show()


    # fig, ax = plot_k_L_E(energy_array, k_L_E_array)
    # # image_path = image_path.parent / f'loglin_{phase[0]:.4f}_{phase[1]:.4f}.png'
    
    # ax.set_yscale('linear')
    # ax.set_ylim(0, np.max([np.max(total_k_L_E_array)*1.25, 5e-9]))
    # for l, en, ra in get_L_label_coords(energy_array, k_L_E_array):
    #     ax.text(en, ra + (ax.get_ylim()[1]-ax.get_ylim()[0])*0.02, f'{l}', fontsize = 'large', color = mpl.colormaps['cividis'](l/30), fontweight = 'bold', va = 'center', ha = 'center')
    # plt.show()
    # # fig.savefig(image_path)

    # # plt.close()

if __name__ == '__main__':
    main()
