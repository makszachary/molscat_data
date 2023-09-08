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

def plotColorMapAndSectionstoFigs(fig0, fig1, singlet_phases: float | np.ndarray[float], triplet_phases: float | np.ndarray[float], phase_differences: float | np.ndarray[float], phase_difference_distinguished: float, so_scaling: float, energy_tuple: tuple[float, ...], temperatures: tuple[float, ...] = (5e-4,), plot_temperature: float = 5e-4, input_dir_name: str = 'RbSr+_tcpld_80mK_0.01_step', hybrid = False, plot_section_lines = False):
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

    ## CONTOUR MAP
    fig0_ax, fig0_ax_bar = ContourMap._initiate_axes(fig0)
    fig0, fig0_ax, fig0_ax_bar, fig0_bar = ContourMap.plotToFigure(fig0, fig0_ax, fig0_ax_bar, singlet_phases, triplet_phases, theory, n_levels = 3)
    fig0_ax.set_xlabel(f'$\\Phi_\\mathrm{{s}}$')
    fig0_ax.set_ylabel(f'$\\Phi_\\mathrm{{t}}$')#, rotation = 0, lapelpad = 12)

    fig0_bar.ax.axhspan((experiment-std)[0], (experiment+std)[0], color = '0.8', alpha=0.8)
    fig0_bar.ax.axhline(experiment[0], color = '1.0', linestyle = '-', linewidth = 2)
    for tick, ticklabel in zip(fig0_bar.ax.get_yticks(), fig0_bar.ax.get_yticklabels()):
        if np.abs(tick - experiment) < 0.05:
            print(f'{tick=}')
            plt.setp(ticklabel, visible=False)
    fig0_bar.ax.text(1.25, experiment, f'$p_\\mathrm{{eff}}^\\mathrm{{exp}}$', va = 'center', ha = 'left')

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
        
        labelLines(lines, align = True, outline_width=2, fontsize = 'small',)
        labelLines(lines, align = True, outline_color=None, fontsize = 'small',)

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

    gs = gridspec.GridSpec(2,90, fig1)
    fig1_ax0 = fig1.add_subplot(gs[0,:-5])
    fig1_ax1 = fig1.add_subplot(gs[1,:-5], sharex = fig1_ax0)

    ### Plot sections for a single temperature but a few values of the phase difference

    T_index = np.nonzero(temperatures == plot_temperature)[0][0]
    theory = arrays_cold_lower[:,:,T_index,0]
    theory_distinguished = np.moveaxis(np.array( [ arrays_cold_lower_distinguished[:,T_index, 0], ]), 0, -1)

    color_map = cmcrameri.cm.devon
    theory_colors = list(reversed([color_map(phase_difference) for phase_difference in phase_differences]))
    theory_formattings = [ {'color': color, 'linewidth': 1.5} for color in theory_colors ]
    # theory_distinguished_colors = ['k', ]
    theory_distinguished_formattings = [ {'color': 'k', 'linewidth': 4, 'linestyle':  '-' } for exp in experiment]
    fig1_ax0 = ValuesVsModelParameters.plotValuestoAxis(fig1_ax0, singlet_phases, theory, experiment, std, theory_distinguished, theory_formattings, theory_distinguished_formattings)

    fig1_ax0.set_xlim(0,1)
    PhaseTicks.setInMultiplesOfPhi(fig1_ax0.xaxis)
    PhaseTicks.linearStr(fig1_ax0.yaxis, 0.2, 0.1, '${x:.1f}$')
    # labelLines(ax0.get_lines(), align = False, outline_width=2, fontsize = 'small', color = 'white')
    # labelLines(ax0.get_lines(), align = False, outline_width=2, outline_color = None, yoffsets= -6.7e-3*(ax0.get_ylim()[1]-ax0.get_ylim()[0]), fontsize = 'small')
    for i, phase_difference in enumerate(phase_differences):
        labelLine(fig1_ax0.get_lines()[i], 0.35, label = f'$\\Delta\\Phi = {phase_difference:.2f}\\pi$', align = False, yoffset = 0.02, outline_width = 2, color = 'white', fontsize = 'small')
        labelLine(fig1_ax0.get_lines()[i], 0.35, label = f'$\\Delta\\Phi = {phase_difference:.2f}\\pi$', align = False, yoffset = 0.02-6.7e-3*(fig1_ax0.get_ylim()[1]-fig1_ax0.get_ylim()[0]), outline_color = None, fontsize = 'small')
    labelLine(fig1_ax0.get_lines()[-1], 0.35, label = f'$\\Delta\\Phi_\\mathrm{{fit}} = {phase_difference_distinguished:.2f}\\pi$', align = False, yoffset = 0.02, outline_width = 2, color = 'white', fontsize = 'small')
    labelLine(fig1_ax0.get_lines()[-1], 0.35, label = f'$\\Delta\\Phi_\\mathrm{{fit}} = {phase_difference_distinguished:.2f}\\pi$', align = False, yoffset = 0.02-6.7e-3*(fig1_ax0.get_ylim()[1]-fig1_ax0.get_ylim()[0]), outline_color = None, fontsize = 'small')
    ### Plot sections for the fitted value of the phase difference but many temperatures
    
    color_map = cmocean.cm.thermal
    lognorm = matplotlib.colors.LogNorm(vmin=min(temperatures), vmax=max(temperatures), clip = False)
    theory_colors = [color_map(lognorm(temperature)) for temperature in temperatures[::2]]
    theory_formattings = [ {'color': color, 'linewidth': 1.5} for color in theory_colors ]
    theory_distinguished_formattings = [ {'color': 'k', 'linewidth': 4, 'linestyle':  (1,(0.1,2)), 'dash_capstyle': 'round' } for exp in experiment]

    T_index = np.nonzero(temperatures == plot_temperature)[0][0]    
    theory = np.moveaxis(arrays_cold_lower_distinguished[:,::2,0], 1, -1)
    theory_distinguished = np.moveaxis(np.array( [ arrays_cold_lower_distinguished[:,T_index, 0], ]), 0, -1)

    fig1_ax1 = ValuesVsModelParameters.plotValuestoAxis(fig1_ax1, singlet_phases, theory, experiment, std, theory_distinguished, theory_formattings, theory_distinguished_formattings)
    PhaseTicks.linearStr(fig1_ax1.yaxis, 0.2, 0.1, '${x:.1f}$')

    # draw the label for the experimental value in the upper plot
    fig1_ax0_right = fig1_ax0.twinx()
    fig1_ax0_right.set_ylim(fig1_ax0.get_ylim())
    fig1_ax0_right.set_yticks( experiment, [f'$p_\\mathrm{{eff}}^\\mathrm{{exp}}$',] )
    fig1_ax0_right.tick_params(axis = 'y', which = 'both', direction = 'in', right = True, length = 10)

    ### Turn off the x-tick labels in the upper plot
    plt.setp(fig1_ax0.get_xticklabels(), visible=False)
    yticks = fig1_ax1.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)

    ### Set the labels
    fig1_ax1.set_xlabel(f'$\\Phi_\\mathrm{{s}}$')
    fig1_ax0.set_ylabel(f'$p_\mathrm{{eff}}$')#, rotation = 0, lapelpad = 12)
    fig1_ax1.set_ylabel(f'$p_\mathrm{{eff}}$')#, rotation = 0, lapelpad = 12)

    ### Set the grid so that the scale on both subplots is not deformed
    lim0 = fig1_ax0.get_ylim()
    lim1 = fig1_ax1.get_ylim()

    gs = gridspec.GridSpec(int(1000*((lim0[1]-lim0[0])+(lim1[1]-lim1[0]))),90)
    fig1_ax0.set_position(gs[0:int(1000*(lim0[1]-lim0[0])),:-5].get_position(fig1))
    fig1_ax0.set_subplotspec(gs[0:int(1000*(lim0[1]-lim0[0])),:-5])
    fig1_ax1.set_position(gs[int(1000*(lim0[1]-lim0[0])):,:-5].get_position(fig1))
    fig1_ax1.set_subplotspec(gs[int(1000*(lim0[1]-lim0[0])):,:-5])

    ### Add the axis for the temperature bar
    fig1_ax1_bar = fig1.add_subplot(gs[int(1000*(lim0[1]-lim0[0])):,-4:])

    
    ### create the temperature bar
    bar_format = theory_distinguished_formattings[0].copy()

    fig1_bar = matplotlib.colorbar.ColorbarBase(fig1_ax1_bar, cmap = color_map, norm = lognorm, ticks = [1e-4, plot_temperature, 1e-3, 1e-2], )
    fig1_bar.set_ticklabels(['$0.1$', f'$T_\\mathrm{{exp}}$', '$1$', '$10$'])
    fig1_bar.ax.axhline(plot_temperature, **bar_format)
    fig1_ax1_bar.tick_params(axis = 'both')
    fig1_ax1_bar.get_yaxis().labelpad = 4
    fig1_ax1_bar.set_ylabel('$T\\,(\\mathrm{mK})$', rotation = 0)
    fig1_ax1_bar.yaxis.set_label_coords(2.1, 1.2)

    return fig0, fig0_ax, fig0_ax_bar, fig0_bar, fig0, fig1, fig1_ax0, fig1_ax0_right, fig1_ax1, fig1_ax1_bar, fig1_bar

def plotMagneticFieldtoFigs(fig2, fig3, magnetic_phases: tuple[tuple[float, float], ...], magnetic_fields: float | np.ndarray[float], magnetic_field_experimental: float, MF_in: int, MS_in: int, energy_tuple: tuple[float, ...], temperatures: tuple[float, ...] = (5e-4,), plot_temperature: float = 5e-4, input_dir_name: str = 'RbSr+_fmf_SE_vs_B_80mK', enhanced = False):
    ## (c) Spin-exchange probabilities vs the magnetic field
    nenergies = len(energy_tuple)
    E_min = min(energy_tuple)
    E_max = max(energy_tuple)
    probabilities_dir_name = 'probabilities'
    prefix_for_array_path = '' if enhanced else 'p0_'
    F1, F2 = 2, 1
    MF1, MF2 = MF_in, MS_in

    abbreviation='cold'
    
    array_paths_cold_lower = [  [arrays_dir_path / input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{triplet_phase:.4f}' / f'{F1}_{MF1}_{F2}_{MF2}' / f'{magnetic_field:.2f}' / probabilities_dir_name / f'{prefix_for_array_path}{abbreviation}.txt' for magnetic_field in magnetic_fields] for singlet_phase, triplet_phase in magnetic_phases]
    [ [print(array_path) for array_path in sublist if (array_path is not None and not array_path.is_file())] for sublist in array_paths_cold_lower ]
    arrays_cold_lower = np.array([ [np.loadtxt(array_path) if (array_path is not None and array_path.is_file()) else np.full((len(temperatures), 3), np.nan) for array_path in sublist] for sublist in array_paths_cold_lower ])
    arrays_cold_lower = arrays_cold_lower.reshape(*arrays_cold_lower.shape[:2], len(temperatures), -1)

    exp_cold_lower = np.loadtxt(data_dir_path / 'exp_data' / 'single_ion_cold_lower.dat')

    peff_experiment = np.array([exp_cold_lower[0,0],])
    peff_std_experiment = np.array([exp_cold_lower[1,0],])
    dpeff = 1e-3
    p0_std = (p0(peff_experiment+dpeff/2, pmf_array=pmf_array)-p0(peff_experiment-dpeff/2, pmf_array=pmf_array))/dpeff * peff_std_experiment
    experiment = peff_experiment if enhanced else p0(peff_experiment, pmf_array)
    std = peff_std_experiment if enhanced else p0_std

    T_index = np.nonzero(temperatures == plot_temperature)[0][0]
    theory = np.moveaxis( arrays_cold_lower[:,:,T_index,0], 0, -1)
    theory_distinguished = None

    color_map = cmcrameri.cm.devon
    theory_colors = list(reversed([color_map(singlet_phase) for singlet_phase, triplet_phase in magnetic_phases]))
    theory_formattings = [ {'color': color, 'linewidth': 2} for color in theory_colors ]
    theory_distinguished_formattings = [ {'color': 'k', 'linewidth': 4, 'linestyle':  (0.8,(0.1,2)), 'dash_capstyle': 'round' } for exp in experiment]

    fig2_ax = fig2.add_subplot()

    fig2_ax = ValuesVsModelParameters.plotValuestoAxis(fig2_ax, magnetic_fields, theory, experiment=None, std=None, theory_distinguished=None, theory_formattings = theory_formattings, theory_distinguished_formattings=theory_distinguished_formattings)
    fig2_ax.scatter([magnetic_field_experimental,], experiment, s = 16, c = theory_distinguished_formattings[0]['color'], marker = 'd')
    fig2_ax.errorbar([magnetic_field_experimental, ], experiment, std, ecolor = theory_distinguished_formattings[0]['color'], capsize = 6)
    fig2_ax.set_ylim(0, 1.05*fig2_ax.get_ylim()[1])
    PhaseTicks.linearStr(fig2_ax.yaxis, 0.1, 0.05, '${x:.1f}$')
    PhaseTicks.linearStr(fig2_ax.xaxis, 50, 10, '${x:n}$')
    for i, (singlet_phase, triplet_phase) in enumerate(magnetic_phases):
        fig2_ax.get_lines()[i].set_label(f'$\\Phi_\\mathrm{{s}} = {singlet_phase:.2f}\\pi$')
    labelLines(fig2_ax.get_lines(), align = False, outline_width=2, fontsize = 'small', color = 'white')
    labelLines(fig2_ax.get_lines(), align = False, outline_width=2, outline_color = None, yoffsets= -6.7e-3*(fig2_ax.get_ylim()[1]-fig2_ax.get_ylim()[0]), fontsize = 'small')
    props = dict(boxstyle='round', facecolor='none', edgecolor='midnightblue')
    fig2_ax.text(0.03, 0.10, f'$\\Delta\\Phi_\\mathrm{{fit}} = {(magnetic_phases[0][1]-magnetic_phases[0][0])%1:.2f}\\pi$', va = 'center', ha = 'left', transform = fig2_ax.transAxes, bbox = props)
    ylabel = f'$p_\mathrm{{eff}}$' if enhanced else f'$p_0$'
    fig2_ax.set_ylabel(ylabel)

    fig2_ax.set_xlabel(f'$B\\,(\\mathrm{{G}})$')

    color_map = cmocean.cm.thermal
    lognorm = matplotlib.colors.LogNorm(vmin=min(temperatures), vmax=max(temperatures), clip = False)
    theory_colors = [color_map(lognorm(temperature)) for temperature in temperatures[::2]]
    theory_formattings = [ {'color': color, 'linewidth': 1.5} for color in theory_colors ]

    gs3 = gridspec.GridSpec(3,60, fig3)
    gs3.update(hspace=0.0)
    fig3_axs = [fig3.add_subplot(gs3[i,:-5], sharex = fig2_ax) for i in range(3)]
    
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

    return fig2, fig2_ax, fig3, fig3_axs

def plotFig3(singlet_phases: float | np.ndarray[float], triplet_phases: float | np.ndarray[float], phase_differences: float | np.ndarray[float], phase_difference_distinguished: float, so_scaling: float, magnetic_phases: tuple[tuple[float, float], ...], magnetic_fields: float | np.ndarray[float], magnetic_field_experimental: float, MF_in: int, MS_in: int, energy_tuple: tuple[float, ...], temperatures: tuple[float, ...] = (5e-4,), plot_temperature: float = 5e-4, cm_input_dir_name: str = 'RbSr+_tcpld_80mK_0.01_step', vs_B_input_dir_name = 'RbSr+_fmf_vs_SE_80mK', colormap_hybrid = False, magnetic_enhanced = False, plot_section_lines = False, journal_name = 'NatCommun'):
    plt.style.use(Path(__file__).parent / 'mpl_style_sheets' / f'{journal_name}.mplstyle')
    nenergies = len(energy_tuple)
    E_min = min(energy_tuple)
    E_max = max(energy_tuple)
    suffix = '_hybrid' if colormap_hybrid else ''
    png_path = plots_dir_path / 'paper' / f'{journal_name}' / 'Fig3' / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'Fig3_{plot_temperature:.2e}K.png'
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
    dpi = 1000
    fig = plt.figure(figsize=figsize, dpi = dpi)
    gs_Figure = gridspec.GridSpec(int(1000*total_height),180, fig)
    # figs = fig.subfigures(2, 2, wspace = ws, hspace = hs)
    fig0 = fig.add_subfigure(gs_Figure[:int(1000*first_row_height),:90])
    fig1 = fig.add_subfigure(gs_Figure[:int(1000*first_row_height),90:])
    fig2 = fig.add_subfigure(gs_Figure[-int(1000*second_row_height):,:120])
    fig3 = fig.add_subfigure(gs_Figure[-int(1000*second_row_height):,120:])

    fig0, fig0_ax, fig0_ax_bar, fig0_bar, fig0, fig1, fig1_ax0, fig1_ax0_right, fig1_ax1, fig1_ax1_bar, fig1_bar = plotColorMapAndSectionstoFigs(fig0, fig1, singlet_phases, triplet_phases, phase_differences, phase_difference_distinguished, so_scaling, energy_tuple, temperatures, plot_temperature, cm_input_dir_name, hybrid = colormap_hybrid, plot_section_lines = plot_section_lines)
    fig2, fig2_ax, fig3, fig3_axs = plotMagneticFieldtoFigs(fig2, fig3, magnetic_phases, magnetic_fields, magnetic_field_experimental, MF_in, MS_in, energy_tuple, temperatures, plot_temperature, vs_B_input_dir_name, enhanced = magnetic_enhanced)

    fig0_ax.text(0., 1.0, f'a', fontsize = 7, family = 'sans-serif', va = 'top', ha = 'left', transform = fig.transFigure, fontweight = 'bold')
    fig1_ax0.text(0.5, 1.00, f'b', fontsize = 7, family = 'sans-serif', va = 'top', ha = 'left', transform = fig.transFigure, fontweight = 'bold')
    fig2_ax.text(0., second_row_height/total_height, f'c', fontsize = 7, family = 'sans-serif', va = 'top', ha = 'left', transform = fig.transFigure, fontweight = 'bold')
    fig3_axs[0].text(0.67, second_row_height/total_height, f'd', fontsize = 7, family = 'sans-serif', va = 'top', ha = 'left', transform = fig.transFigure, fontweight = 'bold')

    fig0.subplots_adjust(left = 0.1*120/90)
    fig1.subplots_adjust(left = 0.17, right = 0.97, hspace = .0)
    fig2.subplots_adjust(left = 0.1, right = 0.97)
    fig3.subplots_adjust(left = 0.17, right = 1-(0.03)*90/60)

    fig.savefig(png_path, bbox_inches='tight', pad_inches = 0)
    fig.savefig(svg_path, bbox_inches='tight', pad_inches = 0, transparent = True)
    fig.savefig(pdf_path, bbox_inches='tight', pad_inches = 0, transparent = True)
    plt.close()

def main():
    parser_description = "This is a python script for running molscat, collecting and pickling S-matrices, and calculating effective probabilities."
    parser = argparse.ArgumentParser(description=parser_description)
    parser.add_argument("-d", "--phase_step", type = float, default = None, help = "The singlet phase step multiples of pi.")
    parser.add_argument("--phase_differences", nargs='*', type = float, default = None, help = "The values of the singlet-triplet semiclassical phase difference modulo pi in multiples of pi.")
    parser.add_argument("--phase_difference", type = float, default = 0.19, help = "The distinguished value of the singlet-triplet semiclassical phase difference modulo pi in multiples of pi.")
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
    parser.add_argument("--cm_input_dir_name", type = str, default = 'RbSr+_tcpld_80mK_0.01_step', help = "Name of the directory with the molscat inputs")
    parser.add_argument("--vs_B_input_dir_name", type = str, default = 'RbSr+_fmf_vs_SE_80mK', help = "Name of the directory with the molscat inputs")
    parser.add_argument("--colormap_hybrid", action = 'store_true', help = "If enabled, the probabilities for the colormap will be taken from 'probabilities_hybrid' directories.")
    parser.add_argument("--plot_section_lines", action = 'store_true', help = "If enabled, the section line for the distinguished phase difference will be drawn.")
    parser.add_argument("--journal", type = str, default = 'NatCommun', help = "Name of the journal to prepare the plots for.")
    args = parser.parse_args()

    nenergies, E_min, E_max, n = args.nenergies, args.E_min, args.E_max, args.n_grid
    energy_tuple = tuple( round(n_root_scale(i, E_min, E_max, nenergies-1, n = n), sigfigs = 11) for i in range(nenergies) )

    # args.phase_difference_step = args.phase_step if args.triplet_phase_step is None else args.triplet_phase_step

    singlet_phases = np.array([default_singlet_phase_function(1.0),]) if args.phase_step is None else np.arange(args.phase_step, 1., args.phase_step).round(decimals=4)
    triplet_phases = np.array([default_triplet_phase_function(1.0),]) if args.phase_step is None else np.arange(args.phase_step, 1., args.phase_step).round(decimals=4)
    phase_differences = np.array(args.phase_differences) if args.phase_differences is not None else np.arange(0., 0.41, 0.1).round(decimals=4)
    so_scaling_values = (0.375,)

    F1, MF1, F2, MF2 = 2, args.MF_in, 1, args.MS_in
    magnetic_phases = tuple(zip(list(args.singlet_phases), list(args.triplet_phases)))
    magnetic_fields = np.arange(args.B_min, args.B_max+0.1*args.dB, args.dB)

    if args.temperatures is None:
        temperatures = list(np.logspace(-4, -2, 20))
        temperatures.append(5e-4)
        temperatures = np.array(sorted(temperatures))
    else:
        temperatures = np.array(args.temperatures)

    [plotFig3(singlet_phases = singlet_phases, triplet_phases = triplet_phases, phase_differences = phase_differences, phase_difference_distinguished = args.phase_difference, so_scaling = so_scaling_values[0], magnetic_phases = magnetic_phases, magnetic_fields = magnetic_fields, magnetic_field_experimental = 2.97, MF_in = MF1, MS_in = MF2, energy_tuple = energy_tuple, temperatures = temperatures, plot_temperature = temperature, cm_input_dir_name = args.cm_input_dir_name, vs_B_input_dir_name = args.vs_B_input_dir_name, colormap_hybrid = args.colormap_hybrid, plot_section_lines = args.plot_section_lines, journal_name = args.journal) for temperature in temperatures]

if __name__ == '__main__':
    main()