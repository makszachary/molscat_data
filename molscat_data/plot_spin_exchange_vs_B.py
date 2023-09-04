import os
from pathlib import Path
import argparse

from multiprocessing import Pool

import numpy as np
from sigfig import round

import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'
from matplotlib import pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

# import palettable
import cmcrameri
import cmocean
from labellines import labelLines, labelLine

import time

from _molscat_data.thermal_averaging import n_root_scale
from _molscat_data.scaling_old import parameter_from_semiclassical_phase, semiclassical_phase_function, default_singlet_parameter_from_phase, default_triplet_parameter_from_phase, default_singlet_phase_function, default_triplet_phase_function
from _molscat_data.effective_probability import effective_probability, p0
from _molscat_data.visualize import ContourMap, ValuesVsModelParameters, PhaseTicks

scratch_path = Path(os.path.expandvars('$SCRATCH'))

data_dir_path = Path(__file__).parents[1] / 'data'
pickles_dir_path = scratch_path / 'python' / 'molscat_data' / 'data_produced' / 'pickles'
pickles_dir_path.mkdir(parents=True, exist_ok=True)
arrays_dir_path = pickles_dir_path.parent / 'arrays'
arrays_dir_path.mkdir(parents=True, exist_ok=True)
plots_dir_path = scratch_path / 'python' / 'molscat_data' / 'plots'

def plot_probability_vs_B(phases: tuple[tuple[float, float], ...], phases_distinguished: tuple[float, float], magnetic_fields: float | np.ndarray[float], magnetic_field_experimental: float, MF_in: int, MS_in: int, energy_tuple: tuple[float, ...], temperatures: tuple[float, ...] = (5e-4,), plot_temperature: float = 5e-4, input_dir_name: str = 'RbSr+_fmf_SE_vs_B_80mK', enhancement = False):
    print(list(locals()))
    nenergies = len(energy_tuple)
    E_min = min(energy_tuple)
    E_max = max(energy_tuple)
    probabilities_dir_name = 'probabilities'
    prefix = '' if enhancement else 'p0_'
    F1, F2 = 2, 1
    MF1, MF2 = MF_in, MS_in

    # array_paths_hot = [ [arrays_dir_path / input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{(singlet_phase+phase_difference)%1:.4f}' / f'{so_scaling:.4f}' / probabilities_dir_name / 'hpf.txt' if ( singlet_phase+phase_difference ) % 1 !=0 else None for phase_difference in phase_differences ] for singlet_phase in singlet_phases]
    # arrays_hot = np.array([ [np.loadtxt(array_path) if (array_path is not None and array_path.is_file()) else np.full((len(temperatures), 5), np.nan) for array_path in sublist] for sublist in array_paths_hot ])
    # arrays_hot = arrays_hot.reshape(*arrays_hot.shape[0:2], len(temperatures), -1)

    # array_paths_cold_higher = [  [arrays_dir_path / input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{(singlet_phase+phase_difference)%1:.4f}' / f'{so_scaling:.4f}' / probabilities_dir_name / 'cold_higher.txt' if ( singlet_phase+phase_difference ) % 1 !=0 else None for phase_difference in phase_differences] for singlet_phase in singlet_phases]
    # arrays_cold_higher = np.array([ [np.loadtxt(array_path) if (array_path is not None and array_path.is_file()) else np.full((len(temperatures), 5), np.nan) for array_path in sublist] for sublist in array_paths_cold_higher ])
    # arrays_cold_higher = arrays_cold_higher.reshape(*arrays_cold_higher.shape[0:2], len(temperatures), -1)

    abbreviation='cold'
    
    array_paths_cold_lower = [  [arrays_dir_path / input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{triplet_phase:.4f}' / f'{F1}_{MF1}_{F2}_{MF2}' / f'{magnetic_field:.2f}' / probabilities_dir_name / f'{abbreviation}.txt' for magnetic_field in magnetic_fields] for singlet_phase, triplet_phase in phases]
    [ [print(array_path) for array_path in sublist if (array_path is not None and not array_path.is_file())] for sublist in array_paths_cold_lower ]
    arrays_cold_lower = np.array([ [np.loadtxt(array_path) if (array_path is not None and array_path.is_file()) else np.full((len(temperatures), 3), np.nan) for array_path in sublist] for sublist in array_paths_cold_lower ])
    arrays_cold_lower = arrays_cold_lower.reshape(*arrays_cold_lower.shape[:2], len(temperatures), -1)

    if phases_distinguished is not None:
        array_paths_cold_lower_distinguished = [arrays_dir_path / input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{phases_distinguished[0]:.4f}_{phases_distinguished[1]:.4f}' / f'{F1}_{MF1}_{F2}_{MF2}' / f'{magnetic_field:.2f}' / probabilities_dir_name / f'{abbreviation}.txt' for magnetic_field in magnetic_fields]
        arrays_cold_lower_distinguished = np.array([ np.loadtxt(array_path) if (array_path is not None and array_path.is_file()) else np.full((len(temperatures), 3), np.nan) for array_path in array_paths_cold_lower_distinguished ])
        arrays_cold_lower_distinguished = arrays_cold_lower_distinguished.reshape(arrays_cold_lower_distinguished.shape[0], len(temperatures), -1)

    # exp_hot = np.loadtxt(data_dir_path / 'exp_data' / 'single_ion_hpf.dat')
    # exp_cold_higher = np.loadtxt(data_dir_path / 'exp_data' / 'single_ion_cold_higher.dat')
    exp_cold_lower = np.loadtxt(data_dir_path / 'exp_data' / 'single_ion_cold_lower.dat')

    experiment = [exp_cold_lower[0,0],]
    std = [exp_cold_lower[1,0],]

    T_index = np.nonzero(temperatures == plot_temperature)[0][0]
    theory = arrays_cold_lower[:,:,T_index,0]
    theory_distinguished = np.moveaxis(np.array( [ arrays_cold_lower_distinguished[:,T_index, 0], ]), 0, -1)
    # theory_distinguished = None

    prefix_for_path = 'peff_' if enhancement else 'p0_'
    png_path = plots_dir_path / 'paper' / f'{prefix_for_path}f=1_SE_vs_B' / f'{input_dir_name}' / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'SE_peff_vs_B_{plot_temperature:.2e}K.png'
    svg_path = png_path.with_suffix('.svg')
    png_path.parent.mkdir(parents = True, exist_ok = True)

    # color_map = matplotlib.colormaps['twilight']
    color_map = cmcrameri.cm.devon
    theory_colors = list(reversed([color_map(phase_difference) for phase_difference in phase_differences]))
    theory_distinguished_colors = ['k', ]

    cm = 1/2.54
    figsize = (9*cm, 7.5*cm)
    dpi = 1000
    fig, ax0 = ValuesVsModelParameters.plotValues(singlet_phases, theory, experiment, std, theory_distinguished, theory_colors, theory_distinguished_colors, figsize=figsize, dpi=dpi)
    ax0.set_xlim(0,1)
    PhaseTicks.setInMultiplesOfPhi(ax0.xaxis)
    for i, (singlet_phase, triplet_phase) in enumerate(phases):
        labelLine(ax0.get_lines()[i], 0.2, label = f'$\\Delta\\Phi = {triplet_phase-singlet_phase:.2f}\\pi,\\, \\Phi_\\mathrm{{s}} = {singlet_phase:.2f}\\pi$', align = False, yoffset = 0.01, outline_width = 4, fontsize = 9, ha = 'left')
    labelLine(ax0.get_lines()[-1], 0.2, label = f'$\\Delta\\Phi = {triplet_phase-singlet_phase:.2f}\\pi,\\, \\Phi_\\mathrm{{s}} = {singlet_phase:.2f}\\pi$', align = False, yoffset = 0.04, outline_width = 4, fontsize = 9, ha = 'left')

    # color_map = matplotlib.colormaps['plasma'] or 'inferno'
    color_map = cmocean.cm.thermal
    lognorm = matplotlib.colors.LogNorm(vmin=min(temperatures), vmax=max(temperatures), clip = False)
    theory_colors = [color_map(lognorm(temperature)) for temperature in temperatures]
    theory_distinguished_colors = ['k', ]

    T_index = np.nonzero(temperatures == plot_temperature)[0][0]    
    theory = np.moveaxis(arrays_cold_lower_distinguished[:,:,0], 1, -1)
    theory_distinguished = np.moveaxis(np.array( [ arrays_cold_lower_distinguished[:,T_index, 0], ]), 0, -1)

    gs = gridspec.GridSpec(2,100)
    ax0.set_position(gs[0,:95].get_position(fig))
    ax0.set_subplotspec(gs[0,:95])

    ax1 = fig.add_subplot(gs[1,:95], sharex = ax0)
    ax1 = ValuesVsModelParameters.plotValuestoAxis(ax1, singlet_phases, theory, experiment, std, theory_distinguished, theory_colors, theory_distinguished_colors)

    lim0 = ax0.get_ylim()
    lim1 = ax1.get_ylim()

    gs = gridspec.GridSpec(int(1000*((lim0[1]-lim0[0])+(lim1[1]-lim1[0]))),100)
    ax0.set_position(gs[0:int(1000*(lim0[1]-lim0[0])),:95].get_position(fig))
    ax0.set_subplotspec(gs[0:int(1000*(lim0[1]-lim0[0])),:95])
    ax1.set_position(gs[int(1000*(lim0[1]-lim0[0]))+1:,:95].get_position(fig))
    ax1.set_subplotspec(gs[int(1000*(lim0[1]-lim0[0]))+1:,:95])
    
    ax1_bar = fig.add_subplot()
    ax1_bar.set_position(gs[int(1000*(lim0[1]-lim0[0]))+1:,96:].get_position(fig))
    ax1_bar.set_subplotspec(gs[int(1000*(lim0[1]-lim0[0]))+1:,96:])

    # draw the label for the experimental value in the upper plot
    # ax0_right = ax0.twinx()
    # ax0_right.set_ylim(ax0.get_ylim())
    # ylabel =f'$p_\\mathrm{{eff}}^\\mathrm{{exp}}$' if enhancement else f'$p_\\mathrm{{0}}^\\mathrm{{exp}}$'
    # ax0_right.set_yticks( experiment, [ylabel,], fontsize = 11 )
    # ax0_right.tick_params(axis = 'y', which = 'both', direction = 'in', right = True, length = 10, labelsize = 11)


    plt.setp(ax0.get_xticklabels(), visible=False)
    yticks = ax1.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)
    
    ax1.set_xlabel(f'$B\\,(\\mathrm{{G}})$', fontsize = 14)
    ylabel = f'$p_\mathrm{{eff}}$' if enhancement else f'$p_0$'
    ax0.set_ylabel(ylabel, fontsize = 14)#, rotation = 0, lapelpad = 12)
    ax1.set_ylabel(ylabel, fontsize = 14)#, rotation = 0, lapelpad = 12)

    # create the temperature bar
    bar = matplotlib.colorbar.ColorbarBase(ax1_bar, cmap = color_map, norm = lognorm, ticks = [1e-4, plot_temperature, 1e-3, 1e-2], )
    bar.set_ticklabels(['$0.1$', f'$T_\\mathrm{{exp}}$', '$1$', '$10$'])
    bar.ax.axhline(plot_temperature, color = '0.', linestyle = '-', linewidth = 4)
    ax1_bar.tick_params(axis = 'both', labelsize = 10)
    ax1_bar.get_yaxis().labelpad = 4
    ax1_bar.set_ylabel('$T\\,(\\mathrm{mK})$', rotation = 0, fontsize = 10)
    ax1_bar.yaxis.set_label_coords(2.1, 1.2)

    # fig.tight_layout()
    fig.subplots_adjust(left = 0.15, top = 0.98, right = 0.9, bottom = 0.15, hspace = .0)
    fig.savefig(png_path, bbox_inches='tight', pad_inches = 0)
    fig.savefig(svg_path, bbox_inches='tight', pad_inches = 0)
    plt.close()

def main():
    parser_description = "This is a python script for running molscat, collecting and pickling S-matrices, and calculating effective probabilities."
    parser = argparse.ArgumentParser(description=parser_description)
    parser.add_argument("-s", "--singlet_phases", nargs='*', type = list, default = [0.04,], help = "The singlet semiclassical phase modulo pi in multiples of pi.")
    parser.add_argument("-t", "--triplet_phases", nargs='*', type = list, default = [0.23,], help = "The triplet semiclassical phase modulo pi in multiples of pi.")
    parser.add_argument("--MF_in", type = int, default = -2)
    parser.add_argument("--MS_in", type = int, default = 1)   
    parser.add_argument("--B_min", type = float, default = 1.0)
    parser.add_argument("--B_max", type = float, default = 100.0)
    parser.add_argument("--dB", type = float, default = 1.0)
    parser.add_argument("--nenergies", type = int, default = 100, help = "Number of energy values in a grid.")
    parser.add_argument("--E_min", type = float, default = 8e-7, help = "Lowest energy value in the grid.")
    parser.add_argument("--E_max", type = float, default = 8e-2, help = "Highest energy value in the grid.")
    parser.add_argument("--n_grid", type = int, default = 3, help = "n parameter for the nth-root energy grid.")
    parser.add_argument("-T", "--temperatures", nargs='*', type = float, default = None, help = "Temperature in the Maxwell-Boltzmann distributions (in kelvins).")
    parser.add_argument("--input_dir_name", type = str, default = 'RbSr+_fmf_SE_vs_B_80mK', help = "Name of the directory with the molscat inputs")
    args = parser.parse_args()

    F1, MF1, F2, MF2 = 2, args.MF_in, 1, args.MS_in

    nenergies, E_min, E_max, n = args.nenergies, args.E_min, args.E_max, args.n_grid
    energy_tuple = tuple( round(n_root_scale(i, E_min, E_max, nenergies-1, n = n), sigfigs = 11) for i in range(nenergies) )
    phases = zip(args.singlet_phases, args.triplet_phases)
    magnetic_fields = np.arange(args.B_min, args.B_max+0.1*args.dB, args.dB)

    if args.temperatures is None:
        temperatures = list(np.logspace(-4, -2, 20))
        temperatures.append(5e-4)
        temperatures = np.array(sorted(temperatures))
    else:
        temperatures = np.array(args.temperatures)

    [plot_probability_vs_B(phases = phases, phases_distinguished= (0.04, 0.23), magnetic_fields = magnetic_fields, magnetic_field_experimental = 3., MF_in = MF1, MS_in = MF2, energy_tuple = energy_tuple, temperatures = temperatures, plot_temperature = temperature, input_dir_name = args.input_dir_name, enhancement = True) for temperature in temperatures]

if __name__ == '__main__':
    main()