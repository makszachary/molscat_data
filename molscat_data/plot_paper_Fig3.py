import sys
import os
from pathlib import Path
import argparse

from multiprocessing import Pool

import numpy as np
from sigfig import round

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import gridspec, ticker
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
if sys.platform == 'win32':
    scratch_path = Path(__file__).parents[3]

data_dir_path = Path(__file__).parents[1] / 'data'
pickles_dir_path = scratch_path / 'python' / 'molscat_data' / 'data_produced' / 'pickles'
pickles_dir_path.mkdir(parents=True, exist_ok=True)
arrays_dir_path = pickles_dir_path.parent / 'arrays'
arrays_dir_path.mkdir(parents=True, exist_ok=True)
plots_dir_path = scratch_path / 'python' / 'molscat_data' / 'plots'

pmf_path = data_dir_path / 'pmf' / 'N_pdf_logic_params_EMM_500uK.txt'
pmf_array = np.loadtxt(pmf_path)

def plotColorMapAndSectionstoFigs(fig0, fig1, phase_step_cm: float, phase_step_sections: float, phase_differences: float | np.ndarray[float], phase_difference_distinguished: float, so_scaling: float, energy_tuple: tuple[float, ...], temperatures: tuple[float, ...] = (5e-4,), plot_temperature: float = 5e-4, input_dir_name: str = 'RbSr+_tcpld_80mK_0.01_step', hybrid = False, plot_section_lines = False, plot_p0 = False, fmf_colormap = False, plot_nan = False,):
    nenergies = len(energy_tuple)
    E_min = min(energy_tuple)
    E_max = max(energy_tuple)
    # singlet_phases, triplet_phases = np.array(singlet_phases), np.array(triplet_phases)
    probabilities_dir_name = 'probabilities_hybrid' if hybrid else 'probabilities'

    F1, F2 = 2, 1
    MF1, MF2 = -2, 1

    singlet_phases_cm = np.array([default_singlet_phase_function(1.0),]) if phase_step_cm is None else np.arange(phase_step_cm, 1., phase_step_cm).round(decimals=4)
    triplet_phases_cm = np.array([default_triplet_phase_function(1.0),]) if phase_step_cm is None else np.arange(phase_step_cm, 1., phase_step_cm).round(decimals=4)

    if fmf_colormap:
        # print('FUCK YOU')
        array_paths_cold_lower = [  [arrays_dir_path / input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{triplet_phase:.4f}' / f'{so_scaling:.4f}' / f'in_{F1}_{MF1}_{F2}_{MF2}' / probabilities_dir_name / ('p0_cold_lower.txt' if plot_p0 else 'cold_lower.txt') if triplet_phase % 1 != 0 else None for triplet_phase in triplet_phases_cm] for singlet_phase in singlet_phases_cm]
        [ [print(f'{array_path} is not a file!') for array_path in sublist if not array_path.is_file()] for sublist in array_paths_cold_lower ]
        # print([ [np.loadtxt(array_path).shape if (array_path is not None and array_path.is_file()) else np.full((len(temperatures), 1), np.nan).shape for array_path in sublist] for sublist in array_paths_cold_lower ])
        arrays_cold_lower = np.array([ [np.loadtxt(array_path) if (array_path is not None and array_path.is_file()) else np.full(temperatures, np.nan) for array_path in sublist] for sublist in array_paths_cold_lower ])
        arrays_cold_lower = arrays_cold_lower.reshape(*arrays_cold_lower.shape[0:2], len(temperatures), -1)
        # print(arrays_cold_lower.shape)
    else:
        array_paths_cold_lower = [  [arrays_dir_path / input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{triplet_phase:.4f}' / f'{so_scaling:.4f}' / probabilities_dir_name / ('p0_cold_lower.txt' if plot_p0 else 'cold_lower.txt') if triplet_phase % 1 != 0 else None for triplet_phase in triplet_phases_cm] for singlet_phase in singlet_phases_cm]
        [ [print(f'{array_path} is not a file!') for array_path in sublist if not array_path.is_file()] for sublist in array_paths_cold_lower ]
        arrays_cold_lower = np.array([ [np.loadtxt(array_path) if (array_path is not None and array_path.is_file()) else np.full((len(temperatures), 3), np.nan) for array_path in sublist] for sublist in array_paths_cold_lower ])
        arrays_cold_lower = arrays_cold_lower.reshape(*arrays_cold_lower.shape[0:2], len(temperatures), -1)

    # exp_hot = np.loadtxt(data_dir_path / 'exp_data' / 'single_ion_hpf.dat')
    # exp_cold_higher = np.loadtxt(data_dir_path / 'exp_data' / 'single_ion_cold_higher.dat')
    exp_cold_lower = np.loadtxt(data_dir_path / 'exp_data' / ('p0_single_ion_cold_lower.dat' if plot_p0 else 'single_ion_cold_lower.dat'))

    experiment = np.array([exp_cold_lower[0,0],])
    std = np.array([exp_cold_lower[1,0],])

    T_index = np.nonzero(temperatures == plot_temperature)[0][0]
    theory = arrays_cold_lower[:,:,T_index,0]

    ## CONTOUR MAP
    fig0_ax, fig0_ax_bar = ContourMap._initiate_axes(fig0)
    fig0, fig0_ax, fig0_ax_bar, fig0_bar = ContourMap.plotToFigure(fig0, fig0_ax, fig0_ax_bar, singlet_phases_cm, triplet_phases_cm, theory, n_levels = 3)
    fig0_ax.set_xlabel(f'$\\Phi_\\mathrm{{s}}$')
    fig0_ax.set_ylabel(f'$\\Phi_\\mathrm{{t}}$')#, rotation = 0, lapelpad = 12)

    fig0_bar.ax.axhspan((experiment-std)[0], (experiment+std)[0], color = '0.8', alpha=0.8)
    fig0_bar.ax.axhline(experiment[0], color = '1.0', linestyle = '-', linewidth = 2)
    for tick, ticklabel in zip(fig0_bar.ax.get_yticks(), fig0_bar.ax.get_yticklabels()):
        if np.abs(tick - experiment) < 0.05:
            print(f'{tick=}')
            plt.setp(ticklabel, visible=False)
    fig0_bar.ax.text(1.25, experiment, (f'$p_0^\\mathrm{{exp}}$' if plot_p0 else f'$p_\\mathrm{{eff}}^\\mathrm{{exp}}$'), va = 'center', ha = 'left', fontsize = matplotlib.rcParams["xtick.labelsize"],)
    if plot_p0: fig0_bar.ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('${x:.2f}$'))

    ### plotting the section lines and their labels
    if plot_section_lines:
        section_distinguished_x = [[min(singlet_phases_cm), max(triplet_phases_cm)-phase_difference_distinguished],
                                    [(min(triplet_phases_cm) - phase_difference_distinguished ) % 1, max(singlet_phases_cm)]
                                    ]
        section_distinguished_y = [[min(singlet_phases_cm)+phase_difference_distinguished, max(triplet_phases_cm)],
                                   [min(triplet_phases_cm), (max(singlet_phases_cm) + phase_difference_distinguished) % 1 ]
                                   ]
        # color_map = cmcrameri.cm.devon
        # theory_colors = list(reversed([color_map(phase_difference) for phase_difference in section_phase_differences]))
        theory_distinguished_colors = ['k', 'k']
        lines  = fig0_ax.plot(section_distinguished_x[0], section_distinguished_y[0], color = theory_distinguished_colors[0], linestyle = 'dashed', linewidth = 1, label = f'$\\Delta\\Phi_\\mathrm{{fit}} = {phase_difference_distinguished:.2f}\\pi$')
        fig0_ax.plot(section_distinguished_x[1], section_distinguished_y[1], color = theory_distinguished_colors[1], linestyle = 'dashed', linewidth = 1, label = f'$\\Delta\\Phi_\\mathrm{{fit}} = {phase_difference_distinguished:.2f}\\pi$')
        
        labelLines(lines, align = True, outline_width=2, color = 'white', fontsize = matplotlib.rcParams["xtick.labelsize"], zorder = 3)
        labelLines(lines, align = True, outline_color=None, fontsize = matplotlib.rcParams["xtick.labelsize"], zorder = 3)

    ## SECTIONS THROUGH THE CONTOUR MAP

    singlet_phases_sections = np.array([default_singlet_phase_function(1.0),]) if phase_step_sections is None else np.arange(phase_step_sections, 1., phase_step_sections).round(decimals=4)

    if fmf_colormap:
        array_paths_cold_lower = [  [arrays_dir_path / input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{(singlet_phase+phase_difference)%1:.4f}' / f'{so_scaling:.4f}' / f'in_{F1}_{MF1}_{F2}_{MF2}' / probabilities_dir_name / ('p0_cold_lower.txt' if plot_p0 else 'cold_lower.txt') if ( singlet_phase+phase_difference ) % 1 !=0 else None for phase_difference in phase_differences] for singlet_phase in singlet_phases_sections]
        [ [print(array_path) for array_path in sublist if (array_path is not None and not array_path.is_file())] for sublist in array_paths_cold_lower ]
        # print([ [np.loadtxt(array_path).shape if (array_path is not None and array_path.is_file()) else np.full(len(temperatures), np.nan).shape for array_path in sublist] for sublist in array_paths_cold_lower ])
        arrays_cold_lower = np.array([ [np.loadtxt(array_path) if (array_path is not None and array_path.is_file()) else np.full(len(temperatures), np.nan) for array_path in sublist] for sublist in array_paths_cold_lower ])
        arrays_cold_lower = arrays_cold_lower.reshape(*arrays_cold_lower.shape[:2], len(temperatures), -1)
    else:
        array_paths_cold_lower = [  [arrays_dir_path / input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{(singlet_phase+phase_difference)%1:.4f}' / f'{so_scaling:.4f}' / probabilities_dir_name / ('p0_cold_lower.txt' if plot_p0 else 'cold_lower.txt') if ( singlet_phase+phase_difference ) % 1 !=0 else None for phase_difference in phase_differences] for singlet_phase in singlet_phases_sections]
        [ [print(array_path) for array_path in sublist if (array_path is not None and not array_path.is_file())] for sublist in array_paths_cold_lower ]
        arrays_cold_lower = np.array([ [np.loadtxt(array_path) if (array_path is not None and array_path.is_file()) else np.full((len(temperatures), 3), np.nan) for array_path in sublist] for sublist in array_paths_cold_lower ])
        arrays_cold_lower = arrays_cold_lower.reshape(*arrays_cold_lower.shape[:2], len(temperatures), -1)

    if phase_difference_distinguished is not None:
        if fmf_colormap:
            array_paths_cold_lower_distinguished = [arrays_dir_path / input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{(singlet_phase+phase_difference_distinguished)%1:.4f}' / f'{so_scaling:.4f}' / f'in_{F1}_{MF1}_{F2}_{MF2}' / probabilities_dir_name / ('p0_cold_lower.txt' if plot_p0 else 'cold_lower.txt') if ( singlet_phase + phase_difference_distinguished ) % 1 !=0 else None for singlet_phase in singlet_phases_sections]
            arrays_cold_lower_distinguished = np.array([ np.loadtxt(array_path) if (array_path is not None and array_path.is_file()) else np.full(len(temperatures), np.nan) for array_path in array_paths_cold_lower_distinguished ])
            arrays_cold_lower_distinguished = arrays_cold_lower_distinguished.reshape(arrays_cold_lower_distinguished.shape[0], len(temperatures), -1)
        else:
            array_paths_cold_lower_distinguished = [arrays_dir_path / input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{(singlet_phase+phase_difference_distinguished)%1:.4f}' / f'{so_scaling:.4f}' / probabilities_dir_name / ('p0_cold_lower.txt' if plot_p0 else 'cold_lower.txt') if ( singlet_phase + phase_difference_distinguished ) % 1 !=0 else None for singlet_phase in singlet_phases_sections]
            arrays_cold_lower_distinguished = np.array([ np.loadtxt(array_path) if (array_path is not None and array_path.is_file()) else np.full((len(temperatures), 3), np.nan) for array_path in array_paths_cold_lower_distinguished ])
            arrays_cold_lower_distinguished = arrays_cold_lower_distinguished.reshape(arrays_cold_lower_distinguished.shape[0], len(temperatures), -1)


    fig1_ax0 = fig1.add_subplot()
    fig1_ax1 = fig1.add_subplot(sharex = fig1_ax0)

    ### Plot sections for a single temperature but a few values of the phase difference

    T_index = np.nonzero(temperatures == plot_temperature)[0][0]
    theory = arrays_cold_lower[:,:,T_index,0]
    if plot_nan:
        theory[np.isnan(theory)] = (theory[np.roll(np.isnan(theory),-1,0)]+theory[np.roll(np.isnan(theory),1,0)])/2
    theory_distinguished = np.moveaxis(np.array( [ arrays_cold_lower_distinguished[:,T_index, 0], ]), 0, -1)

    color_map = cmcrameri.cm.devon
    theory_colors = list(reversed([color_map(phase_difference) for phase_difference in phase_differences]))
    theory_formattings = [ {'color': color, 'linewidth': 1.5} for color in theory_colors ]
    theory_distinguished_formattings = [ {'color': 'k', 'linewidth': 4, 'linestyle':  '-' } for exp in experiment]
    fig1_ax0 = ValuesVsModelParameters.plotValuestoAxis(fig1_ax0, singlet_phases_sections, theory, experiment, std, theory_distinguished, theory_formattings, theory_distinguished_formattings)
    fig1_ax0.set_ylim(0, fig1_ax0.get_ylim()[1])
    fig1_ax0.set_xlim(0,1)
    PhaseTicks.setInMultiplesOfPhi(fig1_ax0.xaxis)
    PhaseTicks.linearStr(fig1_ax0.yaxis, 0.1 if plot_p0 else 0.2, 0.05 if plot_p0 else 0.1, '${x:.1f}$')
    # labelLines(ax0.get_lines(), align = False, outline_width=2, fontsize = matplotlib.rcParams["xtick.labelsize"], color = 'white')
    # labelLines(ax0.get_lines(), align = False, outline_width=2, outline_color = None, yoffsets= -6.7e-3*(ax0.get_ylim()[1]-ax0.get_ylim()[0]), fontsize = matplotlib.rcParams["xtick.labelsize"])
    for i, phase_difference in enumerate(phase_differences):
        labelLine(fig1_ax0.get_lines()[i], 0.35, label = f'$\\Delta\\Phi = {phase_difference:.2f}\\pi$', align = False, yoffset = 0.02, outline_width = 2, color = 'white', fontsize = matplotlib.rcParams["xtick.labelsize"], )
        labelLine(fig1_ax0.get_lines()[i], 0.35, label = f'$\\Delta\\Phi = {phase_difference:.2f}\\pi$', align = False, yoffset = 0.02-6.7e-3*(fig1_ax0.get_ylim()[1]-fig1_ax0.get_ylim()[0]), outline_color = None, fontsize = matplotlib.rcParams["xtick.labelsize"], )
    labelLine(fig1_ax0.get_lines()[-1], 0.35, label = f'$\\Delta\\Phi_\\mathrm{{fit}} = {phase_difference_distinguished:.2f}\\pi$', align = False, yoffset = 0.02, outline_width = 2, color = 'white', fontsize = matplotlib.rcParams["xtick.labelsize"], )
    labelLine(fig1_ax0.get_lines()[-1], 0.35, label = f'$\\Delta\\Phi_\\mathrm{{fit}} = {phase_difference_distinguished:.2f}\\pi$', align = False, yoffset = 0.02-6.7e-3*(fig1_ax0.get_ylim()[1]-fig1_ax0.get_ylim()[0]), outline_color = None, fontsize = matplotlib.rcParams["xtick.labelsize"], )
    ### Plot sections for the fitted value of the phase difference but many temperatures
    
    color_map = cmocean.cm.thermal
    lognorm = matplotlib.colors.LogNorm(vmin=min(temperatures), vmax=max(temperatures), clip = False)
    theory_colors = [color_map(lognorm(temperature)) for temperature in temperatures[::2]]
    theory_formattings = [ {'color': color, 'linewidth': 1.5} for color in theory_colors ]
    theory_distinguished_formattings = [ {'color': 'k', 'linewidth': 4, 'linestyle':  (1.05,(0.1,2)), 'dash_capstyle': 'round' } for exp in experiment]

    T_index = np.nonzero(temperatures == plot_temperature)[0][0]
    # print(f'{arrays_cold_lower_distinguished =}')
    # print(f'{arrays_cold_lower_distinguished.shape =}')
    theory = np.moveaxis(arrays_cold_lower_distinguished[:,::2,0], 1, -1)
    if plot_nan:
        theory[np.isnan(theory)] = (theory[np.roll(np.isnan(theory),-1,0)]+theory[np.roll(np.isnan(theory),1,0)])/2
    theory_distinguished = np.moveaxis(np.array( [ arrays_cold_lower_distinguished[:,T_index, 0], ]), 0, -1)
    # print(f'{theory = }')
    fig1_ax1 = ValuesVsModelParameters.plotValuestoAxis(fig1_ax1, singlet_phases_sections, theory, experiment, std, theory_distinguished, theory_formattings, theory_distinguished_formattings)
    PhaseTicks.linearStr(fig1_ax1.yaxis, 0.1 if plot_p0 else 0.2, 0.05 if plot_p0 else 0.1, '${x:.1f}$')
    fig1_ax1.set_ylim(0, fig1_ax1.get_ylim()[1])

    # draw the label for the experimental value in the upper plot
    fig1_ax0_right = fig1_ax0.twinx()
    fig1_ax0_right.set_ylim(fig1_ax0.get_ylim())
    # fig1_ax0_right.set_yticks(fig1_ax0.get_yticks(minor=True))
    fig1_ax0_right.set_yticks( experiment, [(f'$p_0^\\mathrm{{exp}}$' if plot_p0 else f'$p_\\mathrm{{eff}}^\\mathrm{{exp}}$'),] )
    fig1_ax0_right.tick_params(axis = 'y', which = 'both', direction = 'in', right = True, length = 8)

    ### Turn off the x-tick labels in the upper plot
    plt.setp(fig1_ax0.get_xticklabels(), visible=False)
    yticks = fig1_ax1.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)

    ### Set the labels
    fig1_ax1.set_xlabel(f'$\\Phi_\\mathrm{{s}}$')
    fig1_ax0.set_ylabel(f'$p_0$' if plot_p0 else f'$p_\\mathrm{{eff}}$')#, rotation = 0, lapelpad = 12)
    fig1_ax1.set_ylabel(f'$p_0$' if plot_p0 else f'$p_\\mathrm{{eff}}$')#, rotation = 0, lapelpad = 12)

    ### Set the grid so that the scale on both subplots is not deformed
    lim0 = fig1_ax0.get_ylim()
    lim1 = fig1_ax1.get_ylim()
    # print(f'{lim1 = }')

    gs1 = gridspec.GridSpec(2, 90, fig1, hspace = .15, wspace = 0., height_ratios = [1, 1])# [(lim0[1]-lim0[0]),(lim1[1]-lim1[0])])
    
    fig1_ax0.set_position(gs1[0,:-5].get_position(fig1))
    fig1_ax0.set_subplotspec(gs1[0,:-5])
    fig1_ax1.set_position(gs1[1,:-5].get_position(fig1))
    fig1_ax1.set_subplotspec(gs1[1,:-5])

    ### Add the axis for the temperature bar
    # fig1_ax1_bar = fig1.add_subplot(gs1[int(1000*(lim0[1]-lim0[0])):,-4:])
    fig1_ax1_bar = fig1.add_subplot(gs1[1:,-4:])

    
    ### create the temperature bar
    bar_format = theory_distinguished_formattings[0].copy()

    fig1_bar = matplotlib.colorbar.ColorbarBase(fig1_ax1_bar, cmap = color_map, norm = lognorm, ticks = [1e-4, plot_temperature, 1e-3, 1e-2], )
    fig1_bar.set_ticklabels(['$0.1$', f'$T_\\mathrm{{exp}}$', '$1$', '$10$'])
    fig1_bar.ax.axhline(plot_temperature, **bar_format)
    fig1_ax1_bar.tick_params(axis = 'both')
    fig1_ax1_bar.get_yaxis().labelpad = 4
    fig1_ax1_bar.set_ylabel('$T\\,(\\mathrm{mK})$', rotation = 0, va = 'baseline', ha = 'left')
    fig1_ax1_bar.yaxis.set_label_coords(0.0, 1.08)

    return fig0, fig0_ax, fig0_ax_bar, fig0_bar, fig0, fig1, fig1_ax0, fig1_ax0_right, fig1_ax1, fig1_ax1_bar, fig1_bar, gs1

def plotMagneticFieldtoFigs(fig2, fig3, magnetic_phases: tuple[tuple[float, float], ...], magnetic_fields: float | np.ndarray[float], magnetic_field_experimental: float, energy_tuple: tuple[float, ...], temperatures: tuple[float, ...] = (5e-4,), plot_temperature: float = 5e-4, input_dir_name: str = 'RbSr+_fmf_SE_vs_B_80mK', plot_p0 = False, so_scaling = None,):
    ## (c) Spin-exchange probabilities vs the magnetic field
    nenergies = len(energy_tuple)
    E_min = min(energy_tuple)
    E_max = max(energy_tuple)
    probabilities_dir_name = 'probabilities'
    prefix_for_array_path = '' if not plot_p0 else 'p0_'
    F1, F2 = 2, 1
    MF1, MF2 = -2, 1

    abbreviation='cold'
    if so_scaling is None:
        array_paths_cold_lower = [  [arrays_dir_path / input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{triplet_phase:.4f}' / f'{F1}_{MF1}_{F2}_{MF2}' / f'{magnetic_field:.2f}' / probabilities_dir_name / f'{prefix_for_array_path}{abbreviation}.txt' for magnetic_field in magnetic_fields] for singlet_phase, triplet_phase in magnetic_phases]
        [ [print(array_path) for array_path in sublist if (array_path is not None and not array_path.is_file())] for sublist in array_paths_cold_lower ]
        arrays_cold_lower = np.array([ [np.loadtxt(array_path) if (array_path is not None and array_path.is_file()) else np.full((len(temperatures), 3), np.nan) for array_path in sublist] for sublist in array_paths_cold_lower ])
        arrays_cold_lower = arrays_cold_lower.reshape(*arrays_cold_lower.shape[:2], len(temperatures), -1)

    else:
        array_paths_cold_lower = [  [arrays_dir_path / input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{triplet_phase:.4f}' / f'{F1}_{MF1}_{F2}_{MF2}' / f'{so_scaling:.4f}' / f'{magnetic_field:.2f}' / probabilities_dir_name / f'{prefix_for_array_path}{abbreviation}.txt' for magnetic_field in magnetic_fields] for singlet_phase, triplet_phase in magnetic_phases]
        [ [print(array_path) for array_path in sublist if (array_path is not None and not array_path.is_file())] for sublist in array_paths_cold_lower ]
        arrays_cold_lower = np.array([ [np.loadtxt(array_path) if (array_path is not None and array_path.is_file()) else np.full((len(temperatures), 3), np.nan) for array_path in sublist] for sublist in array_paths_cold_lower ])
        arrays_cold_lower = arrays_cold_lower.reshape(*arrays_cold_lower.shape[:2], len(temperatures), -1)

    exp_cold_lower = np.loadtxt(data_dir_path / 'exp_data' / ('p0_single_ion_cold_lower.dat' if plot_p0 else 'single_ion_cold_lower.dat'))

    experiment = np.array([exp_cold_lower[0,0],])
    std = np.array([exp_cold_lower[1,0],])

    T_index = np.nonzero(temperatures == plot_temperature)[0][0]
    theory = np.moveaxis( arrays_cold_lower[:,:,T_index,0], 0, -1)
    theory_distinguished = None

    color_map = cmcrameri.cm.devon
    theory_colors = list(reversed([color_map(singlet_phase) for singlet_phase, triplet_phase in magnetic_phases]))
    theory_formattings = [ {'color': color, 'linewidth': 2} for color in theory_colors ]
    theory_distinguished_formattings = [ {'color': 'k', 'linewidth': 4, 'linestyle':  (1.05,(0.1,2)), 'dash_capstyle': 'round' } for exp in experiment]

    fig2_ax = fig2.add_subplot()

    fig2_ax = ValuesVsModelParameters.plotValuestoAxis(fig2_ax, magnetic_fields, theory, experiment=None, std=None, theory_distinguished=None, theory_formattings = theory_formattings, theory_distinguished_formattings=theory_distinguished_formattings)
    fig2_ax.scatter([magnetic_field_experimental,], experiment, s = 16, c = theory_distinguished_formattings[0]['color'], marker = 'd', edgecolors = 'dodgerblue')
    fig2_ax.errorbar([magnetic_field_experimental, ], experiment, std, ecolor = theory_distinguished_formattings[0]['color'], capsize = 6)
    fig2_ax.set_ylim(0, 1.05*fig2_ax.get_ylim()[1])
    PhaseTicks.linearStr(fig2_ax.yaxis, 0.1, 0.05, '${x:.1f}$')
    PhaseTicks.linearStr(fig2_ax.xaxis, 50, 10, '${x:n}$')
    for i, (singlet_phase, triplet_phase) in enumerate(magnetic_phases):
        fig2_ax.get_lines()[i].set_label(f'$\\Phi_\\mathrm{{s}} = {singlet_phase:.2f}\\pi$')
    labelLines(fig2_ax.get_lines(), align = False, outline_width=2, color = 'white', fontsize = matplotlib.rcParams["xtick.labelsize"], )
    labelLines(fig2_ax.get_lines(), align = False, outline_color = None, yoffsets= -6.7e-3*(fig2_ax.get_ylim()[1]-fig2_ax.get_ylim()[0]), fontsize = matplotlib.rcParams["xtick.labelsize"], )
    props = dict(boxstyle='round', facecolor='none', edgecolor='midnightblue')
    fig2_ax.text(0.03, 0.10, f'$\\Delta\\Phi_\\mathrm{{fit}} = {(magnetic_phases[0][1]-magnetic_phases[0][0])%1:.2f}\\pi$', va = 'center', ha = 'left', transform = fig2_ax.transAxes, bbox = props)
    ylabel = f'$p_\\mathrm{{eff}}$' if not plot_p0 else f'$p_0$'
    fig2_ax.set_ylabel(ylabel)

    fig2_ax.set_xlabel(f'$B\\,(\\mathrm{{G}})$')

    color_map = cmocean.cm.thermal
    lognorm = matplotlib.colors.LogNorm(vmin=min(temperatures), vmax=max(temperatures), clip = False)
    theory_colors = [color_map(lognorm(temperature)) for temperature in temperatures[::2]]
    theory_formattings = [ {'color': color, 'linewidth': 1.5} for color in theory_colors ]

    gs3 = gridspec.GridSpec(3,60, fig3)
    gs3.update(hspace=0.0)
    fig3_axs = [fig3.add_subplot(gs3[i,:-5], sharex = fig2_ax) for i in range(len(magnetic_phases))]
    
    for i, ax in enumerate(fig3_axs):
        theory = arrays_cold_lower[i,:,::2,0]
        theory_distinguished = np.moveaxis( np.array( [arrays_cold_lower[i,:,T_index,0],]), 0, -1)
        ax = ValuesVsModelParameters.plotValuestoAxis(ax, magnetic_fields, theory, None, None, theory_distinguished, theory_formattings = theory_formattings, theory_distinguished_formattings=theory_distinguished_formattings)
        ax.set_ylim(0, ax.get_ylim()[1])
        PhaseTicks.linearStr(ax.yaxis, 0.2 if ax.get_ylim()[1] > 0.2 else 0.1, 0.1 if ax.get_ylim()[1] > 0.2 else 0.05, '${x:.1f}$')
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('${x:.1f}$'))
    
    for ax in fig3_axs[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)

    for ax in fig3_axs[1:]:
        ax.yaxis.get_major_ticks()[-1].label1.set_visible(False)

    fig3_axs[-1].set_xlabel(f'$B\\,(\\mathrm{{G}})$')

    ### create the temperature bar
    fig3_bar = fig3.add_subplot(gs3[:,-4:])

    bar_format = theory_distinguished_formattings[0].copy()
    # bar_format['linewidth'] = 1
    # bar_format['linestyle'] =  (1,(0.1,1))

    bar = matplotlib.colorbar.ColorbarBase(fig3_bar, cmap = color_map, norm = lognorm, ticks = [1e-4, plot_temperature, 1e-3, 1e-2], )
    bar.set_ticklabels(['$0.1$', f'$T_\\mathrm{{exp}}$', '$1$', '$10$'])
    bar.ax.axhline(plot_temperature, **bar_format)
    fig3_bar.tick_params(axis = 'both')
    fig3_bar.get_yaxis().labelpad = 4
    fig3_bar.set_ylabel('$T\\,(\\mathrm{mK})$', rotation = 0, va = 'baseline', ha = 'left')
    fig3_bar.yaxis.set_label_coords(0.0, 1.05)

    # fig3.subplots_adjust(left = 0.07, top = 0.9, right = 0.95, bottom = 0.20, hspace = .0)

    return fig2, fig2_ax, fig3, fig3_axs, gs3

def plotFig3(phase_step_cm: float, phase_step_sections: float, phase_differences: float | np.ndarray[float], phase_difference_distinguished: float, so_scaling: float, magnetic_phases: tuple[tuple[float, float], ...], magnetic_fields: float | np.ndarray[float], magnetic_field_experimental: float, MF_in: int, MS_in: int, energy_tuple: tuple[float, ...], temperatures: tuple[float, ...] = (5e-4,), plot_temperature: float = 5e-4, cm_input_dir_name: str = 'RbSr+_tcpld_80mK_0.01_step', vs_B_input_dir_name = 'RbSr+_fmf_vs_SE_80mK', colormap_hybrid = False, plot_p0 = False, plot_section_lines = False, fmf_colormap = False, so_scaling_vs_B = False, plot_nan = False, journal_name = 'NatCommun'):
    plt.style.use(Path(__file__).parent / 'mpl_style_sheets' / f'{journal_name}.mplstyle')
    nenergies = len(energy_tuple)
    E_min = min(energy_tuple)
    E_max = max(energy_tuple)
    suffix = '_hybrid' if colormap_hybrid else ''
    png_path = plots_dir_path / 'paper' / f'{journal_name}' / 'Fig3' / f'{cm_input_dir_name}_{vs_B_input_dir_name}' / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'Fig3_{plot_temperature:.2e}K.png'
    pdf_path = png_path.with_suffix('.pdf')
    svg_path = png_path.with_suffix('.svg')
    png_path.parent.mkdir(parents = True, exist_ok = True)

    cm = 1/2.54
    ws, hs = 0.05, 0.05
    first_row_height = 7.5
    vpad = 1
    second_row_height = 6
    total_height = first_row_height+vpad+second_row_height
    figsize = (18*cm, total_height*cm)
    dpi = 1200
    fig = plt.figure(figsize=figsize, dpi = dpi)
    gs_Figure = gridspec.GridSpec(int(1000*total_height),180, fig)
    # figs = fig.subfigures(2, 2, wspace = ws, hspace = hs)
    fig0 = fig.add_subfigure(gs_Figure[:int(1000*first_row_height),:90])
    fig1 = fig.add_subfigure(gs_Figure[:int(1000*first_row_height),90:])
    fig2 = fig.add_subfigure(gs_Figure[-int(1000*second_row_height):,:120])
    fig3 = fig.add_subfigure(gs_Figure[-int(1000*second_row_height):,120:])

    fig0, fig0_ax, fig0_ax_bar, fig0_bar, fig0, fig1, fig1_ax0, fig1_ax0_right, fig1_ax1, fig1_ax1_bar, fig1_bar, gs1 = plotColorMapAndSectionstoFigs(fig0, fig1, phase_step_cm, phase_step_sections, phase_differences, phase_difference_distinguished, so_scaling, energy_tuple, temperatures, plot_temperature, cm_input_dir_name, hybrid = colormap_hybrid, plot_section_lines = plot_section_lines, plot_p0 = plot_p0, fmf_colormap = fmf_colormap, plot_nan = plot_nan)
    fig2, fig2_ax, fig3, fig3_axs, gs3 = plotMagneticFieldtoFigs(fig2, fig3, magnetic_phases, magnetic_fields, magnetic_field_experimental, energy_tuple, temperatures, plot_temperature, vs_B_input_dir_name, plot_p0 = plot_p0, so_scaling = (so_scaling if so_scaling_vs_B else None))

    fig0_ax.text(0., 1.0, f'a', fontsize = 7, family = 'sans-serif', va = 'top', ha = 'left', transform = fig.transFigure, fontweight = 'bold')
    fig1_ax0.text(0.52, 1.00, f'b', fontsize = 7, family = 'sans-serif', va = 'top', ha = 'left', transform = fig.transFigure, fontweight = 'bold')
    fig1_ax0.text(0.52, 1.00-0.5*(first_row_height/total_height+0.0), f'c', fontsize = 7, family = 'sans-serif', va = 'top', ha = 'left', transform = fig.transFigure, fontweight = 'bold')
    fig2_ax.text(0., second_row_height/total_height, f'd', fontsize = 7, family = 'sans-serif', va = 'top', ha = 'left', transform = fig.transFigure, fontweight = 'bold')
    fig3_axs[0].text(0.67, second_row_height/total_height, f'e', fontsize = 7, family = 'sans-serif', va = 'top', ha = 'left', transform = fig.transFigure, fontweight = 'bold')

    fig0.subplots_adjust(left = 0.05)
    gs1.update(left = 0.17, right = 0.97)
    # fig1.subplots_adjust(left = 0.17, right = 0.97)
    fig2.subplots_adjust(left = 0.1, right = 0.97)
    # fig3.subplots_adjust(left = 0.17, right = 1-(0.03)*90/60)
    gs3.update(left = 0.17, right = 1-(0.03)*90/60)

    fig.savefig(png_path, bbox_inches='tight', pad_inches = 0)
    fig.savefig(svg_path, bbox_inches='tight', pad_inches = 0, transparent = True)
    fig.savefig(pdf_path, bbox_inches='tight', pad_inches = 0, transparent = True)

    plt.close()

def main():
    parser_description = "This is a python script for running molscat, collecting and pickling S-matrices, and calculating effective probabilities."
    parser = argparse.ArgumentParser(description=parser_description)
    parser.add_argument("--phase_step_cm", type = float, default = 0.04, help = "The phase step in the multiples of pi for the color map.")
    parser.add_argument("--phase_step_sections", type = float, default = 0.04, help = "The singlet phase step in themultiples of pi for the sections through the color map.")
    parser.add_argument("--phase_differences", nargs='*', type = float, default = None, help = "The values of the singlet-triplet semiclassical phase difference modulo pi in multiples of pi.")
    parser.add_argument("--phase_difference", type = float, default = 0.19, help = "The distinguished value of the singlet-triplet semiclassical phase difference modulo pi in multiples of pi.")
    parser.add_argument("--so_scaling", type = float, default = 0.375, help = "Value of the SO scaling.")

    parser.add_argument("-s", "--singlet_phases", nargs='*', type = float, default = [0.04,], help = "The singlet semiclassical phase modulo pi in multiples of pi for the plot of magnetic field.")
    parser.add_argument("-t", "--triplet_phases", nargs='*', type = float, default = [0.23,], help = "The triplet semiclassical phase modulo pi in multiples of pi for the plot of magnetic field.")

    parser.add_argument("--MF_in", type = int, default = -2)
    parser.add_argument("--MS_in", type = int, default = 1)   

    parser.add_argument("--B_min", type = float, default = 1.0)
    parser.add_argument("--B_max", type = float, default = 100.0)
    parser.add_argument("--dB", type = float, default = 1.0)

    parser.add_argument("--nenergies", type = int, default = 50, help = "Number of energy values in a grid.")
    parser.add_argument("--E_min", type = float, default = 8e-7, help = "Lowest energy value in the grid.")
    parser.add_argument("--E_max", type = float, default = 8e-2, help = "Highest energy value in the grid.")
    parser.add_argument("--n_grid", type = int, default = 3, help = "n parameter for the nth-root energy grid.")

    parser.add_argument("-T", "--temperatures", nargs='*', type = float, default = None, help = "Temperature in the Maxwell-Boltzmann distributions (in kelvins).")
    parser.add_argument("--nT", type = int, default = 20, help = "Number of temperatures included in the calculations.")
    parser.add_argument("--logT_min", type = float, default = -4)
    parser.add_argument("--logT_max", type = float, default = -2)

    parser.add_argument("--fmf_colormap", action = 'store_true', help = "Assume that the scattering calculations in molscat were done with the fmf basis set. Changes the directory structure for arrays.")
    parser.add_argument("--so_scaling_vs_B", action = 'store_true', help = "Assume that the scattering calculations in molscat were done with spin-orbit coupling included. Changes the directory structure for arrays.")

    parser.add_argument("--cm_input_dir_name", type = str, default = 'RbSr+_tcpld_80mK_0.01_step', help = "Name of the directory with the molscat inputs")
    parser.add_argument("--vs_B_input_dir_name", type = str, default = 'RbSr+_fmf_SE_vs_B_80mK', help = "Name of the directory with the molscat inputs")
    parser.add_argument("--colormap_hybrid", action = 'store_true', help = "If enabled, the probabilities for the colormap will be taken from 'probabilities_hybrid' directories.")
    parser.add_argument("--plot_section_lines", action = 'store_true', help = "If enabled, the section line for the distinguished phase difference will be drawn.")
    parser.add_argument("--plot_p0", action = 'store_true', help = "If included, the short-range probability p0 will be plotted instead of peff.")
    parser.add_argument("--plot_nan", action = 'store_true', help = "If included, the plotted values will be interpolated for arrays that weren't found (instead of jus plotting a blank place).")

    parser.add_argument("--journal", type = str, default = 'NatCommun', help = "Name of the journal to prepare the plots for.")
    args = parser.parse_args()

    nenergies, E_min, E_max, n = args.nenergies, args.E_min, args.E_max, args.n_grid
    energy_tuple = tuple( round(n_root_scale(i, E_min, E_max, nenergies-1, n = n), sigfigs = 11) for i in range(nenergies) )

    # args.phase_difference_step = args.phase_step if args.triplet_phase_step is None else args.triplet_phase_step

    phase_differences = np.array(args.phase_differences) if args.phase_differences is not None else np.arange(0., 0.41, 0.1).round(decimals=4)
    so_scaling = args.so_scaling

    F1, MF1, F2, MF2 = 2, args.MF_in, 1, args.MS_in
    magnetic_phases = tuple(zip(list(args.singlet_phases), list(args.triplet_phases)))
    magnetic_fields = np.arange(args.B_min, args.B_max+0.1*args.dB, args.dB)

    if args.temperatures is None:
        temperatures = list(np.logspace(args.logT_min, args.logT_max, args.nT))
        temperatures.append(5e-4)
        temperatures = np.array(sorted(temperatures))
    else:
        temperatures = np.array(args.temperatures)

    [plotFig3(phase_step_cm = args.phase_step_cm, phase_step_sections = args.phase_step_sections, phase_differences = phase_differences, phase_difference_distinguished = args.phase_difference, so_scaling = so_scaling, magnetic_phases = magnetic_phases, magnetic_fields = magnetic_fields, magnetic_field_experimental = 2.97, MF_in = MF1, MS_in = MF2, energy_tuple = energy_tuple, temperatures = temperatures, plot_temperature = temperature, cm_input_dir_name = args.cm_input_dir_name, vs_B_input_dir_name = args.vs_B_input_dir_name, colormap_hybrid = args.colormap_hybrid, plot_p0 = args.plot_p0, plot_section_lines = args.plot_section_lines, journal_name = args.journal, fmf_colormap = args.fmf_colormap, fmf_vs_B = args.so_scaling_vs_B, plot_nan = args.plot_nan) for temperature in temperatures]

if __name__ == '__main__':
    main()