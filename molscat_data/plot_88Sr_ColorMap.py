import os
from pathlib import Path
import argparse

from multiprocessing import Pool

import numpy as np
from sigfig import round

import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'cm'
from matplotlib import pyplot as plt
import cmcrameri

import time

from _molscat_data.thermal_averaging import n_root_scale
from _molscat_data.scaling_old import parameter_from_semiclassical_phase, semiclassical_phase_function, default_singlet_parameter_from_phase, default_triplet_parameter_from_phase, default_singlet_phase_function, default_triplet_phase_function
from _molscat_data.effective_probability import effective_probability, p0
from _molscat_data.visualize import ContourMap
from _molscat_data.visualize import ValuesVsModelParameters
from labellines import labelLines, labelLine

scratch_path = Path(os.path.expandvars('$SCRATCH'))

data_dir_path = Path(__file__).parents[1] / 'data'
pickles_dir_path = scratch_path / 'python' / 'molscat_data' / 'data_produced' / 'pickles'
pickles_dir_path.mkdir(parents=True, exist_ok=True)
arrays_dir_path = pickles_dir_path.parent / 'arrays'
arrays_dir_path.mkdir(parents=True, exist_ok=True)
plots_dir_path = scratch_path / 'python' / 'molscat_data' / 'plots'

def plotColorMap(singlet_phases: float | np.ndarray[float], triplet_phases: float | np.ndarray[float], so_scaling: float, energy_tuple: tuple[float, ...], temperatures: tuple[float, ...] = (5e-4,), plot_temperature: float = 5e-4, input_dir_name: str = 'RbSr+_tcpld_80mK_0.04_step', hybrid = False, plot_section_lines = False):
    nenergies = len(energy_tuple)
    E_min = min(energy_tuple)
    E_max = max(energy_tuple)
    singlet_phases, triplet_phases = np.array(singlet_phases), np.array(triplet_phases)
    probabilities_dir_name = 'probabilities_hybrid' if hybrid else 'probabilities'

    # array_paths_hot = [ [arrays_dir_path / input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{(singlet_phase+phase_difference)%1:.4f}' / f'{so_scaling:.4f}' / probabilities_dir_name / 'hpf.txt' if ( singlet_phase+phase_difference ) % 1 !=0 else None for phase_difference in phase_differences ] for singlet_phase in singlet_phases]
    # arrays_hot = np.array([ [np.loadtxt(array_path) if (array_path is not None and array_path.is_file()) else np.full((len(temperatures), 5), np.nan) for array_path in sublist] for sublist in array_paths_hot ])
    # arrays_hot = arrays_hot.reshape(*arrays_hot.shape[0:2], len(temperatures), -1)

    # array_paths_cold_higher = [  [arrays_dir_path / input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{(singlet_phase+phase_difference)%1:.4f}' / f'{so_scaling:.4f}' / probabilities_dir_name / 'cold_higher.txt' if ( singlet_phase+phase_difference ) % 1 !=0 else None for phase_difference in phase_differences] for singlet_phase in singlet_phases]
    # arrays_cold_higher = np.array([ [np.loadtxt(array_path) if (array_path is not None and array_path.is_file()) else np.full((len(temperatures), 5), np.nan) for array_path in sublist] for sublist in array_paths_cold_higher ])
    # arrays_cold_higher = arrays_cold_higher.reshape(*arrays_cold_higher.shape[0:2], len(temperatures), -1)

    array_paths_cold_lower = [  [arrays_dir_path / input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{triplet_phase:.4f}' / f'{so_scaling:.4f}' / probabilities_dir_name / 'cold_lower.txt' if triplet_phase % 1 != 0 else None for triplet_phase in triplet_phases] for singlet_phase in singlet_phases]
    [ [print(array_path) for array_path in sublist if not array_path.is_file()] for sublist in array_paths_cold_lower ]
    arrays_cold_lower = np.array([ [np.loadtxt(array_path) if (array_path is not None and array_path.is_file()) else np.full((len(temperatures), 3), np.nan) for array_path in sublist] for sublist in array_paths_cold_lower ])
    arrays_cold_lower = arrays_cold_lower.reshape(*arrays_cold_lower.shape[0:2], len(temperatures), -1)

    # exp_hot = np.loadtxt(data_dir_path / 'exp_data' / 'single_ion_hpf.dat')
    # exp_cold_higher = np.loadtxt(data_dir_path / 'exp_data' / 'single_ion_cold_higher.dat')
    exp_cold_lower = np.loadtxt(data_dir_path / 'exp_data' / 'single_ion_cold_lower.dat')

    experiment = exp_cold_lower[0,0]
    std = exp_cold_lower[1,0]

    T_index = np.nonzero(temperatures == plot_temperature)[0][0]
    theory = arrays_cold_lower[:,:,T_index,0]

    suffix = '_hybrid' if hybrid else ''
    png_path = plots_dir_path / 'paper' / 'ColorMap_f=1_SE' / f'{input_dir_name}{suffix}' / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'SE_peff_ColorMap_{plot_temperature:.2e}K.png'
    pdf_path = plots_dir_path / 'paper' / 'ColorMap_f=1_SE' / f'{input_dir_name}{suffix}' / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'SE_peff_ColorMap_{plot_temperature:.2e}K.pdf'
    svg_path = png_path.with_suffix('.svg')
    png_path.parent.mkdir(parents = True, exist_ok = True)

    cm = 1/2.54
    figsize = (9*cm, 7.5*cm)
    dpi = 1000
    fig, ax, ax_bar, bar = ContourMap.plotMap(singlet_phases, triplet_phases, theory, n_levels=3, figsize = figsize, dpi = dpi)
    ax.set_xlabel(f'$\\Phi_\\mathrm{{s}}$', fontsize = 14)
    ax.set_ylabel(f'$\\Phi_\\mathrm{{t}}$', fontsize = 14)#, rotation = 0, lapelpad = 12)

    bar.ax.axhspan(experiment-std, experiment+std, color = '0.8', alpha=0.8)
    bar.ax.axhline(experiment, color = '1.0', linestyle = '-', linewidth = 2)
    for tick, ticklabel in zip(bar.ax.get_yticks(), bar.ax.get_yticklabels()):
        if np.abs(tick - experiment) < 0.05:
            plt.setp(ticklabel, visible=False)
    bar.ax.text(1.25, experiment, f'$p_\\mathrm{{eff}}^\\mathrm{{exp}}$', fontsize = 11, va = 'center', ha = 'left')

    ## plotting the section lines and their labels
    if plot_section_lines:
        section_phase_differences = [0.0, 0.1, 0.30, 0.40]
        section_lines_x = [ [min(singlet_phases), max(singlet_phases)-phase_difference] for phase_difference in section_phase_differences ]
        section_lines_y = [ [min(singlet_phases)+phase_difference, max(triplet_phases)] for phase_difference in section_phase_differences ]
        section_distinguished_x = [min(singlet_phases), max(singlet_phases)-0.19]
        section_distinguished_y = [min(singlet_phases)+0.19, max(triplet_phases)]
        color_map = cmcrameri.cm.devon
        theory_colors = list(reversed([color_map(phase_difference) for phase_difference in section_phase_differences]))
        theory_distinguished_colors = ['k', ]
        lines = []
        for i, (xx, yy) in enumerate(zip(section_lines_x, section_lines_y)):
            line, =ax.plot(xx, yy, color = theory_colors[i], linestyle = 'dashed', linewidth = .5, label = f'$\\Delta\\Phi = {section_phase_differences[i]:.2f}\\pi$')
            lines.append(line)
        line, = ax.plot(section_distinguished_x, section_distinguished_y, color = theory_distinguished_colors[0], linestyle = 'dashed', linewidth = 1, label = f'$\\Delta\\Phi_\\mathrm{{fit}} = {0.19:.2f}\\pi$')
        lines.append(line)

        labelLines(lines, align = True, outline_width=2, fontsize = 8,)


    fig.subplots_adjust(bottom=0.15)
    fig.savefig(png_path, bbox_inches='tight', pad_inches = 0)
    # fig.savefig(pdf_path, bbox_inches='tight', pad_inches = 0)
    fig.savefig(svg_path, bbox_inches='tight', pad_inches = 0)
    plt.close()

def main():
    parser_description = "This is a python script for running molscat, collecting and pickling S-matrices, and calculating effective probabilities."
    parser = argparse.ArgumentParser(description=parser_description)
    parser.add_argument("-d", "--phase_step", type = float, default = None, help = "The phase step multiples of pi.")
    # parser.add_argument("-s", "--singlet_phase", type = float, default = None, help = "The distinguished value of the singlet semiclassical phase modulo pi in multiples of pi.")
    # parser.add_argument("-t", "--triplet_phase", type = float, default = None, help = "The distinguished value of the triplet semiclassical phase modulo pi in multiples of pi.")
    parser.add_argument("--nenergies", type = int, default = 100, help = "Number of energy values in a grid.")
    parser.add_argument("--E_min", type = float, default = 8e-7, help = "Lowest energy value in the grid.")
    parser.add_argument("--E_max", type = float, default = 8e-2, help = "Highest energy value in the grid.")
    parser.add_argument("--n_grid", type = int, default = 3, help = "n parameter for the nth-root energy grid.")
    parser.add_argument("-T", "--temperatures", nargs='*', type = float, default = None, help = "Temperature in the Maxwell-Boltzmann distributions (in kelvins).")
    parser.add_argument("--input_dir_name", type = str, default = 'RbSr+_tcpld_80mK', help = "Name of the directory with the molscat inputs")
    parser.add_argument("--hybrid", action = 'store_true', help = "If enabled, the probabilities will be taken from 'probabilities_hybrid' directories.")
    parser.add_argument("--plot_section_lines", action = 'store_true', help = "If enabled, the section lines will be drawn.")
    args = parser.parse_args()

    nenergies, E_min, E_max, n = args.nenergies, args.E_min, args.E_max, args.n_grid
    energy_tuple = tuple( round(n_root_scale(i, E_min, E_max, nenergies-1, n = n), sigfigs = 11) for i in range(nenergies) )

    singlet_phases = np.array([default_singlet_phase_function(1.0),]) if args.phase_step is None else np.arange(args.phase_step, 1., args.phase_step).round(decimals=4)
    triplet_phases = np.array([default_triplet_phase_function(1.0),]) if args.phase_step is None else np.arange(args.phase_step, 1., args.phase_step).round(decimals=4)
    so_scaling_values = (0.375,)

    if args.temperatures is None:
        temperatures = list(np.logspace(-4, -2, 20))
        temperatures.append(5e-4)
        temperatures = np.array(sorted(temperatures))
    else:
        temperatures = np.array(args.temperatures)

    [plotColorMap(singlet_phases = singlet_phases, triplet_phases = triplet_phases, so_scaling = so_scaling_values[0], energy_tuple = energy_tuple, temperatures = temperatures, plot_temperature = temperature, input_dir_name = args.input_dir_name, hybrid = args.hybrid, plot_section_lines = args.plot_section_lines) for temperature in temperatures]

if __name__ == '__main__':
    main()