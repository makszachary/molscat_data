import os
from pathlib import Path
import argparse

from multiprocessing import Pool

import numpy as np
from sigfig import round

import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'
from matplotlib import pyplot as plt
# plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
params = {'text.usetex' : True,
          'font.size': 11,
          'axes.labelsize': 11,
          'legend.labelsize': 11,
          'font.family': 'lmodern',
          'text.latex.unicode': True,
          'text.latex.preamble': (
              r'\usepackage{lmodern}'
          )}
plt.rcParams.update(params)
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

def plotPeffVsPhis(singlet_phases: float | np.ndarray[float], phase_differences: float | np.ndarray[float], phase_difference_distinguished: float, so_scaling: float, energy_tuple: tuple[float, ...], temperatures: tuple[float, ...] = (5e-4,), plot_temperature: float = 5e-4, input_dir_name: str = 'RbSr+_tcpld_80mK_0.04_step', hybrid = False):
    nenergies = len(energy_tuple)
    E_min = min(energy_tuple)
    E_max = max(energy_tuple)
    singlet_phases, phase_differences = np.array(singlet_phases), np.array(phase_differences)
    probabilities_dir_name = 'probabilities_hybrid' if hybrid else 'probabilities'

    # array_paths_hot = [ [arrays_dir_path / input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{(singlet_phase+phase_difference)%1:.4f}' / f'{so_scaling:.4f}' / probabilities_dir_name / 'hpf.txt' if ( singlet_phase+phase_difference ) % 1 !=0 else None for phase_difference in phase_differences ] for singlet_phase in singlet_phases]
    # arrays_hot = np.array([ [np.loadtxt(array_path) if (array_path is not None and array_path.is_file()) else np.full((len(temperatures), 5), np.nan) for array_path in sublist] for sublist in array_paths_hot ])
    # arrays_hot = arrays_hot.reshape(*arrays_hot.shape[0:2], len(temperatures), -1)

    # array_paths_cold_higher = [  [arrays_dir_path / input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{(singlet_phase+phase_difference)%1:.4f}' / f'{so_scaling:.4f}' / probabilities_dir_name / 'cold_higher.txt' if ( singlet_phase+phase_difference ) % 1 !=0 else None for phase_difference in phase_differences] for singlet_phase in singlet_phases]
    # arrays_cold_higher = np.array([ [np.loadtxt(array_path) if (array_path is not None and array_path.is_file()) else np.full((len(temperatures), 5), np.nan) for array_path in sublist] for sublist in array_paths_cold_higher ])
    # arrays_cold_higher = arrays_cold_higher.reshape(*arrays_cold_higher.shape[0:2], len(temperatures), -1)

    array_paths_cold_lower = [  [arrays_dir_path / input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{(singlet_phase+phase_difference)%1:.4f}' / f'{so_scaling:.4f}' / probabilities_dir_name / 'cold_lower.txt' if ( singlet_phase+phase_difference ) % 1 !=0 else None for phase_difference in phase_differences] for singlet_phase in singlet_phases]
    [ [print(array_path) for array_path in sublist if (array_path is not None and not array_path.is_file())] for sublist in array_paths_cold_lower ]
    arrays_cold_lower = np.array([ [np.loadtxt(array_path) if (array_path is not None and array_path.is_file()) else np.full((len(temperatures), 3), np.nan) for array_path in sublist] for sublist in array_paths_cold_lower ])
    arrays_cold_lower = arrays_cold_lower.reshape(*arrays_cold_lower.shape[:2], len(temperatures), -1)

    if phase_difference_distinguished is not None:
        array_paths_cold_lower_distinguished = [arrays_dir_path / input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{(singlet_phase+phase_difference_distinguished)%1:.4f}' / f'{so_scaling:.4f}' / probabilities_dir_name / 'cold_lower.txt' if ( singlet_phase + phase_difference_distinguished ) % 1 !=0 else None for singlet_phase in singlet_phases]
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

    suffix = '_hybrid' if hybrid else ''
    png_path = plots_dir_path / 'paper' / 'peff_f=1_SE_vs_Phis_many_DPhi' / f'{input_dir_name}{suffix}' / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'SE_peff_vs_Phis_{plot_temperature:.2e}K.png'
    svg_path = png_path.with_suffix('.svg')
    png_path.parent.mkdir(parents = True, exist_ok = True)

    # color_map = matplotlib.colormaps['twilight']
    color_map = cmcrameri.cm.devon
    theory_colors = list(reversed([color_map(phase_difference) for phase_difference in phase_differences]))
    theory_distinguished_colors = ['k', ]


    figsize = (9.5, 8)
    dpi = 1000
    fig, ax0 = ValuesVsModelParameters.plotValues(singlet_phases, theory, experiment, std, theory_distinguished, theory_colors, theory_distinguished_colors, figsize=figsize, dpi=dpi)
    ax0.set_xlim(0,1)
    PhaseTicks.setInMultiplesOfPhi(ax0.xaxis)
    for i, phase_difference in enumerate(phase_differences):
        labelLine(ax0.get_lines()[i], 0.25, label = f'$\\Delta\\Phi = {phase_difference:.2f}\\pi$', align = False, yoffset = 0.01, outline_width = 4, fontsize = 14)
    labelLine(ax0.get_lines()[-1], 0.25, label = f'$\\Delta\\Phi_\\mathrm{{fit}} = {phase_difference_distinguished:.2f}\\pi$', align = False, yoffset = 0.04, outline_width = 4, fontsize = 14)

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
    ax0_right = ax0.twinx()
    ax0_right.set_ylim(ax0.get_ylim())
    ax0_right.set_yticks( experiment, [f'$p_\\mathrm{{eff}}^\\mathrm{{exp}}$',], fontsize = 16 )
    ax0_right.tick_params(axis = 'y', which = 'both', direction = 'in', right = True, length = 10, labelsize = 16)

    plt.setp(ax0.get_xticklabels(), visible=False)
    yticks = ax1.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)
    
    ax1.set_xlabel(f'$\\Phi_\\mathrm{{s}}$', fontsize = 24)
    ax0.set_ylabel(f'$p_\mathrm{{eff}}$', fontsize = 24)#, rotation = 0, lapelpad = 12)
    ax1.set_ylabel(f'$p_\mathrm{{eff}}$', fontsize = 24)#, rotation = 0, lapelpad = 12)

    # create the temperature bar
    bar = matplotlib.colorbar.ColorbarBase(ax1_bar, cmap = color_map, norm = lognorm, ticks = [1e-4, plot_temperature, 1e-3, 1e-2], )
    bar.set_ticklabels(['$0.1$', f'$T_\\mathrm{{exp}}$', '$1$', '$10$'])
    bar.ax.axhline(plot_temperature, color = '0.', linestyle = '-', linewidth = 4)
    ax1_bar.tick_params(axis = 'both', labelsize = 14)
    ax1_bar.get_yaxis().labelpad = 4
    ax1_bar.set_ylabel('$T\\,(\\mathrm{mK})$', rotation = 0, fontsize = 14)
    ax1_bar.yaxis.set_label_coords(2.1, 1.15)

    # fig.tight_layout()
    fig.subplots_adjust(left = 0.15, top = 0.98, right = 0.9, bottom = 0.15, hspace = .0)
    fig.savefig(png_path, bbox_inches='tight', pad_inches = 0)
    fig.savefig(svg_path, bbox_inches='tight', pad_inches = 0)
    plt.close()

def main():
    parser_description = "This is a python script for running molscat, collecting and pickling S-matrices, and calculating effective probabilities."
    parser = argparse.ArgumentParser(description=parser_description)
    parser.add_argument("-d", "--phase_step", type = float, default = None, help = "The singlet phase step multiples of pi.")
    parser.add_argument("--phase_differences", nargs='*', type = float, default = None, help = "The values of the singlet-triplet semiclassical phase difference modulo pi in multiples of pi.")
    # parser.add_argument("--phase_difference_step", type = float, default = None, help = "The distinguished value of the singlet-triplet semiclassical phase difference modulo pi in multiples of pi.")
    parser.add_argument("--phase_difference", type = float, default = 0.19, help = "The distinguished value of the singlet-triplet semiclassical phase difference modulo pi in multiples of pi.")
    parser.add_argument("--nenergies", type = int, default = 100, help = "Number of energy values in a grid.")
    parser.add_argument("--E_min", type = float, default = 8e-7, help = "Lowest energy value in the grid.")
    parser.add_argument("--E_max", type = float, default = 8e-2, help = "Highest energy value in the grid.")
    parser.add_argument("--n_grid", type = int, default = 3, help = "n parameter for the nth-root energy grid.")
    parser.add_argument("-T", "--temperatures", nargs='*', type = float, default = None, help = "Temperature in the Maxwell-Boltzmann distributions (in kelvins).")
    parser.add_argument("--input_dir_name", type = str, default = 'RbSr+_tcpld_80mK', help = "Name of the directory with the molscat inputs")
    parser.add_argument("--hybrid", action = 'store_true', help = "If enabled, the probabilities will be taken from 'probabilities_hybrid' directories.")
    args = parser.parse_args()

    nenergies, E_min, E_max, n = args.nenergies, args.E_min, args.E_max, args.n_grid
    energy_tuple = tuple( round(n_root_scale(i, E_min, E_max, nenergies-1, n = n), sigfigs = 11) for i in range(nenergies) )

    # args.phase_difference_step = args.phase_step if args.triplet_phase_step is None else args.triplet_phase_step

    singlet_phases = np.array([default_singlet_phase_function(1.0),]) if args.phase_step is None else np.arange(args.phase_step, 1., args.phase_step).round(decimals=4)
    # phase_differences = np.array([args.phase_difference,]) if args.phase_difference_step is None else np.arange(0, 0.7, args.phase_difference_step).round(decimals=4)
    phase_differences = np.array(args.phase_differences) if args.phase_differences is not None else np.arange(0., 0.51, 0.1).round(decimals=4)
    so_scaling_values = (0.375,)

    if args.temperatures is None:
        temperatures = list(np.logspace(-4, -2, 20))
        temperatures.append(5e-4)
        temperatures = np.array(sorted(temperatures))
    else:
        temperatures = np.array(args.temperatures)

    [plotPeffVsPhis(singlet_phases = singlet_phases, phase_differences = phase_differences, phase_difference_distinguished = args.phase_difference, so_scaling = so_scaling_values[0], energy_tuple = energy_tuple, temperatures = temperatures, plot_temperature = temperature, input_dir_name = args.input_dir_name, hybrid = args.hybrid) for temperature in temperatures]

if __name__ == '__main__':
    main()