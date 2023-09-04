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


def plotColorMapAndSections(singlet_phases: float | np.ndarray[float], triplet_phases: float | np.ndarray[float], phase_differences: float | np.ndarray[float], phase_difference_distinguished: float, so_scaling: float, energy_tuple: tuple[float, ...], temperatures: tuple[float, ...] = (5e-4,), plot_temperature: float = 5e-4, input_dir_name: str = 'RbSr+_tcpld_80mK_0.04_step', hybrid = False, plot_section_lines = False):
    nenergies = len(energy_tuple)
    E_min = min(energy_tuple)
    E_max = max(energy_tuple)
    singlet_phases, triplet_phases = np.array(singlet_phases), np.array(triplet_phases)
    probabilities_dir_name = 'probabilities_hybrid' if hybrid else 'probabilities'

    array_paths_cold_lower = [  [arrays_dir_path / input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{triplet_phase:.4f}' / f'{so_scaling:.4f}' / probabilities_dir_name / 'cold_lower.txt' if triplet_phase % 1 != 0 else None for triplet_phase in triplet_phases] for singlet_phase in singlet_phases]
    [ [print(array_path) for array_path in sublist if not array_path.is_file()] for sublist in array_paths_cold_lower ]
    arrays_cold_lower = np.array([ [np.loadtxt(array_path) if (array_path is not None and array_path.is_file()) else np.full((len(temperatures), 3), np.nan) for array_path in sublist] for sublist in array_paths_cold_lower ])
    arrays_cold_lower = arrays_cold_lower.reshape(*arrays_cold_lower.shape[0:2], len(temperatures), -1)

    # exp_hot = np.loadtxt(data_dir_path / 'exp_data' / 'single_ion_hpf.dat')
    # exp_cold_higher = np.loadtxt(data_dir_path / 'exp_data' / 'single_ion_cold_higher.dat')
    exp_cold_lower = np.loadtxt(data_dir_path / 'exp_data' / 'single_ion_cold_lower.dat')

    experiment = np.array([exp_cold_lower[0,0],])
    std = np.array([exp_cold_lower[1,0],])

    T_index = np.nonzero(temperatures == plot_temperature)[0][0]
    theory = arrays_cold_lower[:,:,T_index,0]

    suffix = '_hybrid' if hybrid else ''
    png_path = plots_dir_path / 'paper' / 'ColorMap_f=1_SE_with_sections' / f'{input_dir_name}{suffix}' / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'SE_peff_ColorMap_{plot_temperature:.2e}K.png'
    pdf_path = png_path.with_suffix('.pdf')
    svg_path = png_path.with_suffix('.svg')
    png_path.parent.mkdir(parents = True, exist_ok = True)

    cm = 1/2.54
    figsize = (18.5*cm, 7.5*cm)
    dpi = 1000

    fig = plt.figure(figsize=figsize, dpi = dpi)
    fig0, fig1 = fig.subfigures(1, 2, wspace=0.05,)

    ## CONTOUR MAP
    fig0_ax, fig0_ax_bar = ContourMap._initiate_axes(fig0)
    fig0, fig0_ax, fig0_ax_bar, fig0_bar = ContourMap.plotToFigure(fig0, fig0_ax, fig0_ax_bar, singlet_phases, triplet_phases, theory, n_levels = 3)
    fig0_ax.set_xlabel(f'$\\Phi_\\mathrm{{s}}$', fontsize = 14)
    fig0_ax.set_ylabel(f'$\\Phi_\\mathrm{{t}}$', fontsize = 14)#, rotation = 0, lapelpad = 12)

    fig0_bar.ax.axhspan((experiment-std)[0], (experiment+std)[0], color = '0.8', alpha=0.8)
    fig0_bar.ax.axhline(experiment[0], color = '1.0', linestyle = '-', linewidth = 2)
    for tick, ticklabel in zip(fig0_bar.ax.get_yticks(), fig0_bar.ax.get_yticklabels()):
        if np.abs(tick - experiment) < 0.05:
            print(f'{tick=}')
            plt.setp(ticklabel, visible=False)
    fig0_bar.ax.text(1.25, experiment, f'$p_\\mathrm{{eff}}^\\mathrm{{exp}}$', fontsize = 11, va = 'center', ha = 'left')

    ### plotting the section lines and their labels
    if plot_section_lines:
        section_distinguished_x = [[min(singlet_phases), max(triplet_phases)-phase_difference_distinguished],
                                    [(min(triplet_phases) - phase_difference_distinguished ) % 1, max(singlet_phases)]
                                    ]
        section_distinguished_y = [[min(singlet_phases)+phase_difference_distinguished, max(triplet_phases)],
                                   [min(triplet_phases), (max(singlet_phases) + phase_difference_distinguished) % 1 ]
                                   ]
        # color_map = cmcrameri.cm.devon
        # theory_colors = list(reversed([color_map(phase_difference) for phase_difference in section_phase_differences]))
        theory_distinguished_colors = ['k', 'k']
        lines  = fig0_ax.plot(section_distinguished_x[0], section_distinguished_y[0], color = theory_distinguished_colors[0], linestyle = 'dashed', linewidth = 1, label = f'$\\Delta\\Phi_\\mathrm{{fit}} = {0.19:.2f}\\pi$')
        fig0_ax.plot(section_distinguished_x[1], section_distinguished_y[1], color = theory_distinguished_colors[1], linestyle = 'dashed', linewidth = 1, label = f'$\\Delta\\Phi_\\mathrm{{fit}} = {0.19:.2f}\\pi$')
        
        labelLines(lines, align = True, outline_width=2, fontsize = 8,)

    ## SECTIONS THROUGH THE CONTOUR MAP

    array_paths_cold_lower = [  [arrays_dir_path / input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{(singlet_phase+phase_difference)%1:.4f}' / f'{so_scaling:.4f}' / probabilities_dir_name / 'cold_lower.txt' if ( singlet_phase+phase_difference ) % 1 !=0 else None for phase_difference in phase_differences] for singlet_phase in singlet_phases]
    [ [print(array_path) for array_path in sublist if (array_path is not None and not array_path.is_file())] for sublist in array_paths_cold_lower ]
    arrays_cold_lower = np.array([ [np.loadtxt(array_path) if (array_path is not None and array_path.is_file()) else np.full((len(temperatures), 3), np.nan) for array_path in sublist] for sublist in array_paths_cold_lower ])
    arrays_cold_lower = arrays_cold_lower.reshape(*arrays_cold_lower.shape[:2], len(temperatures), -1)

    if phase_difference_distinguished is not None:
        array_paths_cold_lower_distinguished = [arrays_dir_path / input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{(singlet_phase+phase_difference_distinguished)%1:.4f}' / f'{so_scaling:.4f}' / probabilities_dir_name / 'cold_lower.txt' if ( singlet_phase + phase_difference_distinguished ) % 1 !=0 else None for singlet_phase in singlet_phases]
        arrays_cold_lower_distinguished = np.array([ np.loadtxt(array_path) if (array_path is not None and array_path.is_file()) else np.full((len(temperatures), 3), np.nan) for array_path in array_paths_cold_lower_distinguished ])
        arrays_cold_lower_distinguished = arrays_cold_lower_distinguished.reshape(arrays_cold_lower_distinguished.shape[0], len(temperatures), -1)

    # experiment = [exp_cold_lower[0,0],]
    # std = [exp_cold_lower[1,0],]

    gs = gridspec.GridSpec(2,100, fig1)
    fig1_ax0 = fig1.add_subplot(gs[0,:95])
    fig1_ax1 = fig1.add_subplot(gs[1,:95], sharex = fig1_ax0)

    ### Plot sections for a single temperature but a few values of the phase difference

    T_index = np.nonzero(temperatures == plot_temperature)[0][0]
    theory = arrays_cold_lower[:,:,T_index,0]
    theory_distinguished = np.moveaxis(np.array( [ arrays_cold_lower_distinguished[:,T_index, 0], ]), 0, -1)

    color_map = cmcrameri.cm.devon
    theory_colors = list(reversed([color_map(phase_difference) for phase_difference in phase_differences]))
    theory_distinguished_colors = ['k', ]
    fig1_ax0 = ValuesVsModelParameters.plotValuestoAxis(fig1_ax0, singlet_phases, theory, experiment, std, theory_distinguished, theory_colors, theory_distinguished_colors)

    fig1_ax0.set_xlim(0,1)
    PhaseTicks.setInMultiplesOfPhi(fig1_ax0.xaxis)
    PhaseTicks.linearStr(fig1_ax0.yaxis, 0.2, 0.1, '${x:.1f}$')
    for i, phase_difference in enumerate(phase_differences):
        labelLine(fig1_ax0.get_lines()[i], 0.2, label = f'$\\Delta\\Phi = {phase_difference:.2f}\\pi$', align = False, yoffset = 0.01, outline_width = 4, fontsize = 9, ha = 'left')
    labelLine(fig1_ax0.get_lines()[-1], 0.2, label = f'$\\Delta\\Phi_\\mathrm{{fit}} = {phase_difference_distinguished:.2f}\\pi$', align = False, yoffset = 0.04, outline_width = 4, fontsize = 9, ha = 'left')
    
    ### Plot sections for the fitted value of the phase difference but many temperatures
    
    color_map = cmocean.cm.thermal
    lognorm = matplotlib.colors.LogNorm(vmin=min(temperatures), vmax=max(temperatures), clip = False)
    theory_colors = [color_map(lognorm(temperature)) for temperature in temperatures[::2]]
    theory_distinguished_colors = ['k', ]

    T_index = np.nonzero(temperatures == plot_temperature)[0][0]    
    theory = np.moveaxis(arrays_cold_lower_distinguished[:,::2,0], 1, -1)
    theory_distinguished = np.moveaxis(np.array( [ arrays_cold_lower_distinguished[:,T_index, 0], ]), 0, -1)

    fig1_ax1 = ValuesVsModelParameters.plotValuestoAxis(fig1_ax1, singlet_phases, theory, experiment, std, theory_distinguished, theory_colors, theory_distinguished_colors)
    PhaseTicks.linearStr(fig1_ax1.yaxis, 0.2, 0.1, '${x:.1f}$')

    # draw the label for the experimental value in the upper plot
    fig1_ax0_right = fig1_ax0.twinx()
    fig1_ax0_right.set_ylim(fig1_ax0.get_ylim())
    fig1_ax0_right.set_yticks( experiment, [f'$p_\\mathrm{{eff}}^\\mathrm{{exp}}$',], fontsize = 11 )
    fig1_ax0_right.tick_params(axis = 'y', which = 'both', direction = 'in', right = True, length = 10, labelsize = 11)

    ### Turn off the x-tick labels in the upper plot
    plt.setp(fig1_ax0.get_xticklabels(), visible=False)
    yticks = fig1_ax1.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)

    ### Set the labels
    fig1_ax1.set_xlabel(f'$\\Phi_\\mathrm{{s}}$', fontsize = 14)
    fig1_ax0.set_ylabel(f'$p_\mathrm{{eff}}$', fontsize = 14)#, rotation = 0, lapelpad = 12)
    fig1_ax1.set_ylabel(f'$p_\mathrm{{eff}}$', fontsize = 14)#, rotation = 0, lapelpad = 12)

    ### Set the grid so that the scale on both subplots is not deformed
    lim0 = fig1_ax0.get_ylim()
    lim1 = fig1_ax1.get_ylim()

    gs = gridspec.GridSpec(int(1000*((lim0[1]-lim0[0])+(lim1[1]-lim1[0]))),100)
    fig1_ax0.set_position(gs[0:int(1000*(lim0[1]-lim0[0])),:95].get_position(fig))
    fig1_ax0.set_subplotspec(gs[0:int(1000*(lim0[1]-lim0[0])),:95])
    fig1_ax1.set_position(gs[int(1000*(lim0[1]-lim0[0]))+1:,:95].get_position(fig))
    fig1_ax1.set_subplotspec(gs[int(1000*(lim0[1]-lim0[0]))+1:,:95])

    ### Add the axis for the temperature bar
    fig1_ax1_bar = fig1.add_subplot()
    fig1_ax1_bar.set_position(gs[int(1000*(lim0[1]-lim0[0]))+1:,96:].get_position(fig1))
    fig1_ax1_bar.set_subplotspec(gs[int(1000*(lim0[1]-lim0[0]))+1:,96:])

    
    ### create the temperature bar
    fig1_bar = matplotlib.colorbar.ColorbarBase(fig1_ax1_bar, cmap = color_map, norm = lognorm, ticks = [1e-4, plot_temperature, 1e-3, 1e-2], )
    fig1_bar.set_ticklabels(['$0.1$', f'$T_\\mathrm{{exp}}$', '$1$', '$10$'])
    fig1_bar.ax.axhline(plot_temperature, color = '0.', linestyle = '-', linewidth = 4)
    fig1_ax1_bar.tick_params(axis = 'both', labelsize = 10)
    fig1_ax1_bar.get_yaxis().labelpad = 4
    fig1_ax1_bar.set_ylabel('$T\\,(\\mathrm{mK})$', rotation = 0, fontsize = 10)
    fig1_ax1_bar.yaxis.set_label_coords(2.1, 1.2)

    fig0_ax.text(-0.17, 1.00, f'a', fontsize = 10, va = 'top', ha = 'left', transform = fig0_ax.transAxes, fontweight = 'bold')
    fig1_ax0.text(-0.17, 1.00, f'b', fontsize = 10, va = 'top', ha = 'left', transform = fig1_ax0.transAxes, fontweight = 'bold')
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
    parser.add_argument("--nenergies", type = int, default = 50, help = "Number of energy values in a grid.")
    parser.add_argument("--E_min", type = float, default = 8e-7, help = "Lowest energy value in the grid.")
    parser.add_argument("--E_max", type = float, default = 8e-2, help = "Highest energy value in the grid.")
    parser.add_argument("--n_grid", type = int, default = 3, help = "n parameter for the nth-root energy grid.")
    parser.add_argument("-T", "--temperatures", nargs='*', type = float, default = None, help = "Temperature in the Maxwell-Boltzmann distributions (in kelvins).")
    parser.add_argument("--input_dir_name", type = str, default = 'RbSr+_tcpld_80mK', help = "Name of the directory with the molscat inputs")
    parser.add_argument("--hybrid", action = 'store_true', help = "If enabled, the probabilities will be taken from 'probabilities_hybrid' directories.")
    parser.add_argument("--plot_section_lines", action = 'store_true', help = "If enabled, the section line for the distinguished phase difference will be drawn.")
    args = parser.parse_args()

    nenergies, E_min, E_max, n = args.nenergies, args.E_min, args.E_max, args.n_grid
    energy_tuple = tuple( round(n_root_scale(i, E_min, E_max, nenergies-1, n = n), sigfigs = 11) for i in range(nenergies) )

    # args.phase_difference_step = args.phase_step if args.triplet_phase_step is None else args.triplet_phase_step

    singlet_phases = np.array([default_singlet_phase_function(1.0),]) if args.phase_step is None else np.arange(args.phase_step, 1., args.phase_step).round(decimals=4)
    triplet_phases = np.array([default_triplet_phase_function(1.0),]) if args.phase_step is None else np.arange(args.phase_step, 1., args.phase_step).round(decimals=4)
    phase_differences = np.array(args.phase_differences) if args.phase_differences is not None else np.arange(0., 0.41, 0.1).round(decimals=4)
    so_scaling_values = (0.375,)

    if args.temperatures is None:
        temperatures = list(np.logspace(-4, -2, 20))
        temperatures.append(5e-4)
        temperatures = np.array(sorted(temperatures))
    else:
        temperatures = np.array(args.temperatures)

    [plotColorMapAndSections(singlet_phases = singlet_phases, triplet_phases = triplet_phases, phase_differences = phase_differences, phase_difference_distinguished = args.phase_difference, so_scaling = so_scaling_values[0], energy_tuple = energy_tuple, temperatures = temperatures, plot_temperature = temperature, input_dir_name = args.input_dir_name, hybrid = args.hybrid, plot_section_lines = args.plot_section_lines) for temperature in temperatures]

if __name__ == '__main__':
    main()