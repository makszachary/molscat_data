import sys
import os
from pathlib import Path
import argparse
import zipfile

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

from _molscat_data import quantum_numbers as qn
from _molscat_data.smatrix import SMatrixCollection
from _molscat_data.thermal_averaging import n_root_scale, n_root_iterator
from _molscat_data.scaling_old import parameter_from_semiclassical_phase, semiclassical_phase_function, default_singlet_parameter_from_phase, default_triplet_parameter_from_phase, default_singlet_phase_function, default_triplet_phase_function
from _molscat_data.effective_probability import effective_probability, p0
from _molscat_data.visualize import ContourMap, ValuesVsModelParameters, PhaseTicks, Barplot, BicolorHandler
from _molscat_data.physical_constants import amu_to_au, red_mass_87Rb_84Sr_amu, red_mass_87Rb_86Sr_amu, red_mass_87Rb_87Sr_amu, red_mass_87Rb_88Sr_amu
from _molscat_data.chi_squared import chi_squared

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

def plotFig2(singlet_phase: float, triplet_phase: float, so_scaling: float, reduced_masses: np.ndarray[float], energy_tuple_barplot: tuple[float, ...], energy_tuple_vs_mass_even: tuple[float, ...], energy_tuple_vs_mass_odd: tuple[float, ...], temperatures: np.ndarray[float] = np.array([5e-4,]), plot_temperature: float = 5e-4, barplot_input_dir_name: str = 'RbSr+_tcpld_80mK_0.01_step', barplot_SE_input_dir_name: str = 'RbSr+_tcpld_80mK_0.01_step_SE', vs_mass_even_input_dir_name = 'RbSr+_tcpld_80mK_vs_mass', vs_mass_odd_input_dir_name = 'RbSr+_tcpld_vs_mass_odd', fmf_barplot = False, journal_name = 'NatCommun', plot_p0 = False):
    plt.style.use(Path(__file__).parent / 'mpl_style_sheets' / f'{journal_name}.mplstyle')
    # nenergies = len(energy_tuple_barplot)
    # E_min = min(energy_tuple_barplot)
    # E_max = max(energy_tuple_barplot)
    png_path = plots_dir_path / 'paper' / f'{journal_name}' / 'Fig2' / f'Fig2_{plot_temperature:.2e}K.png'
    pdf_path = png_path.with_suffix('.pdf')
    svg_path = png_path.with_suffix('.svg')
    log_path = png_path.with_suffix('.log')
    data_path = png_path.with_suffix('.txt')
    png_path.parent.mkdir(parents = True, exist_ok = True)
    
    
    cm = 1/2.54
    ws, hs = 0.05, 0.05
    nrows = 3 # 2
    row_height = 5
    vpad = .5
    total_height = nrows*row_height + (nrows-1)*vpad
    figsize = (8.8*cm, total_height*cm)
    dpi = 1200
    fig = plt.figure(figsize = figsize, dpi = dpi)
    gs_Figure = gridspec.GridSpec(nrows, 1, fig, hspace = hs, wspace = ws, height_ratios = [1 for row in range(nrows)])
    # figs = fig.subfigures(2, 2, wspace = ws, hspace = hs)
    figs = [fig.add_subfigure(gs_Figure[i]) for i in range(nrows)]
    # figs[0] = fig.add_subfigure(gs_Figure[0])
    # figs[1] = fig.add_subfigure(gs_Figure[1])
    figs_axes = [[] for fig in figs]
    # fig2 = fig.add_subfigure(gs_Figure[2])
    # fig3 = fig.add_subfigure(gs_Figure[3])
    
    
    nenergies = len(energy_tuple_barplot)
    E_min = min(energy_tuple_barplot)
    E_max = max(energy_tuple_barplot)
    probabilities_dir_name = 'probabilities'

    if fmf_barplot:
        arrays_path_hpf = [arrays_dir_path / barplot_input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{triplet_phase:.4f}' / f'{so_scaling:.4f}' / f'in_4_{MF_in}_1_1' / probabilities_dir_name / ('p0_hpf.txt' if plot_p0 else 'hpf.txt') for MF_in in range(-4, 4+1, 2)]
        arrays_path_cold_higher = [arrays_dir_path / barplot_input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{triplet_phase:.4f}' / f'{so_scaling:.4f}' / f'in_4_{MF_in}_1_1' / probabilities_dir_name / ('p0_cold_higher.txt' if plot_p0 else 'cold_higher.txt') for MF_in in range(-4, 4+1, 2)]
        SE_arrays_path_hpf = [arrays_dir_path / barplot_SE_input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{triplet_phase:.4f}' / f'{0.0:.4f}' / f'in_4_{MF_in}_1_1' / probabilities_dir_name / ('p0_hpf.txt' if plot_p0 else 'hpf.txt') for MF_in in range(-4, 4+1, 2)]
        SE_arrays_path_cold_higher = [arrays_dir_path / barplot_SE_input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{triplet_phase:.4f}' / f'{0.0:.4f}' / f'in_4_{MF_in}_1_1' / probabilities_dir_name / ('p0_cold_higher.txt' if plot_p0 else 'cold_higher.txt') for MF_in in range(-4, 4+1, 2)]
        arrays_hpf = np.array([np.loadtxt(path) for path in arrays_path_hpf]).transpose()
        arrays_cold_higher = np.array([np.loadtxt(path) for path in arrays_path_cold_higher]).transpose()
        SE_arrays_hpf = np.array([np.loadtxt(path) for path in SE_arrays_path_hpf]).transpose()
        SE_arrays_cold_higher = np.array([np.loadtxt(path) for path in SE_arrays_path_cold_higher]).transpose()
    else:
        arrays_path_hpf = arrays_dir_path / barplot_input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{triplet_phase:.4f}' / f'{so_scaling:.4f}' / probabilities_dir_name / ('p0_hpf.txt' if plot_p0 else 'hpf.txt')
        arrays_path_cold_higher = arrays_dir_path / barplot_input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{triplet_phase:.4f}' / f'{so_scaling:.4f}' / probabilities_dir_name / ('p0_cold_higher.txt' if plot_p0 else 'cold_higher.txt')
        SE_arrays_path_hpf = arrays_dir_path / barplot_SE_input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{triplet_phase:.4f}' / probabilities_dir_name / ('p0_hpf.txt' if plot_p0 else 'hpf.txt')
        SE_arrays_path_cold_higher = arrays_dir_path / barplot_SE_input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{triplet_phase:.4f}' / probabilities_dir_name / ('p0_cold_higher.txt' if plot_p0 else 'cold_higher.txt')
        arrays_hpf = np.loadtxt(arrays_path_hpf)
        arrays_cold_higher = np.loadtxt(arrays_path_cold_higher)
        SE_arrays_hpf = np.loadtxt(SE_arrays_path_hpf)
        SE_arrays_cold_higher = np.loadtxt(SE_arrays_path_cold_higher)


    T_index = np.nonzero(temperatures == plot_temperature)[0][0]
    theory_hpf= arrays_hpf[T_index,:]
    theory_SE_hpf = SE_arrays_hpf[T_index,:]
    theory_cold_higher = arrays_cold_higher[T_index,:]
    theory_SE_cold_higher = SE_arrays_cold_higher[T_index,:]

    np.savetxt(data_path.with_stem(data_path.stem+'_hpf'), theory_hpf, fmt = '%.4f')
    np.savetxt(data_path.with_stem(data_path.stem+'_hpf_SE'), theory_SE_hpf, fmt = '%.4f')
    np.savetxt(data_path.with_stem(data_path.stem+'_cold_higher'), theory_cold_higher, fmt = '%.4f')
    np.savetxt(data_path.with_stem(data_path.stem+'_cold_SE_higher'), theory_SE_cold_higher, fmt = '%.4f')
    # print(f'{plot_temperature=}')
    # print(f'{arrays_path_hpf=}')
    # print(f'{theory_hpf=}')
    # print(f'{theory_SE_hpf=}')

    experiment_std_path_hpf = Path(__file__).parents[1] / 'data' / 'exp_data' / ('p0_single_ion_hpf.dat' if plot_p0 else 'single_ion_hpf.dat')
    experiment_std_path_cold_higher = Path(__file__).parents[1] / 'data' / 'exp_data' / ('p0_single_ion_cold_higher.dat' if plot_p0 else 'single_ion_cold_higher.dat')
    experiment_hpf = np.loadtxt(experiment_std_path_hpf)[0]
    std_hpf = np.loadtxt(experiment_std_path_hpf)[1]
    experiment_cold_higher = np.loadtxt(experiment_std_path_cold_higher)[0]
    std_cold_higher = np.loadtxt(experiment_std_path_cold_higher)[1]
    

    f_max = 2
    barplot_labels = [ f'$\\left|{str(int(f_max))},{str(int(mf))}\\right\\rangle$'.replace(f'-', f'{{-}}') for mf in np.arange (-f_max, f_max+1)]
    # barplot_labels = [ f'$\\left|2,-2\\right\\rangle$', f'$\\left|2,{{-}}1\\right\\rangle$', f'$\\left|2,0\\right\\rangle$', f'$\\left|2,1\\right\\rangle$', f'$\\left|2,2\\right\\rangle$',]

    figs_axes[0].append(figs[0].add_subplot())
    bars_formatting_hpf = { 'facecolor': 'indianred', 'edgecolor': 'black', 'alpha': 0.9, 'ecolor': 'black', 'capsize': 3 }
    SE_bars_formatting_hpf = { 'facecolor': 'firebrick', 'edgecolor': 'black', 'alpha': 0.9, 'ecolor': 'black', 'capsize': 3 }
    bars_formatting_cold_higher = { 'facecolor': 'royalblue', 'edgecolor': 'black', 'alpha': 0.9, 'ecolor': 'black', 'capsize': 3 }
    SE_bars_formatting_cold_higher = { 'facecolor': 'midnightblue', 'edgecolor': 'black', 'alpha': 0.9, 'ecolor': 'black', 'capsize': 3 }

    exp_formatting = { 'color': 'orange', 'marker': 'x', 'markersize': 4, 'capsize': 3, 'linestyle': 'none' }
    bars_formatting = [ bars_formatting_hpf, bars_formatting_cold_higher ]
    SE_bars_formatting = [ SE_bars_formatting_hpf, SE_bars_formatting_cold_higher ]

    theory = np.array([ theory_hpf, theory_cold_higher ])
    theory_SE = np.array([ theory_SE_hpf, theory_SE_cold_higher ])
    experiment = np.array([ experiment_hpf, experiment_cold_higher ])
    std = np.array([ std_hpf, std_cold_higher ])

    
    figs_axes[0][0] = Barplot.plotBarplotConciseToAxes(figs_axes[0][0], theory, experiment, std, barplot_labels, theory_SE, bars_formatting = bars_formatting, exp_formatting = exp_formatting, SE_bars_formatting = SE_bars_formatting, )
    number_of_datasets = theory.shape[0] if len(theory.shape) == 2 else 1
    number_of_xticks = theory.shape[1] if len(theory.shape) == 2 else theory.shape[0]
    positions = np.array([ [(number_of_datasets+1)*k+(j+1) for k in range(number_of_xticks)]
                                                            for j in range(number_of_datasets)] )
    indices_used_for_fitting = [(0,0),(1,0),(0,-1)]
    positions_used_for_fitting = [ positions[i] for i in indices_used_for_fitting ]
    theory_used_for_fitting = [ theory[i] for i in indices_used_for_fitting ]
    figs_axes[0][0].bar(positions_used_for_fitting, theory_used_for_fitting, width = 1, facecolor = 'none', hatch = '////', edgecolor = 'k', linewidth = .1)
    PhaseTicks.linearStr(figs_axes[0][0].yaxis, 0.1, 0.05, '${x:.1f}$')
    figs_axes[0][0].set_ylim(0, 0.3)# 1.30*np.amax(theory))

    ylabel = (f'$p_0$' if plot_p0 else f'$p_\\mathrm{{eff}}$')
    figs_axes[0][0].set_ylabel(ylabel)

    labels_and_colors = { 'hyperfine relaxation\n(without & with SO coupling)': SE_bars_formatting_hpf['facecolor'],
                         'cold spin flip\n(without & with SO coupling)': SE_bars_formatting_cold_higher['facecolor'], }
    handles_colors = [ plt.Rectangle((0,0), 1, 1, facecolor = labels_and_colors[color_label], edgecolor = 'k', hatch = '' ) for color_label in labels_and_colors.keys() ]
    colors_and_hatches = [ (SE_format['facecolor'], SO_format['facecolor'], '') for SE_format, SO_format in zip(SE_bars_formatting, bars_formatting, strict = True) ]
    labels = [ *list(labels_and_colors.keys()),]
    
    handles_colors.append(plt.Rectangle((0,0), 1, 1, facecolor = 'none', edgecolor = 'k', hatch = '' ))    
    colors_and_hatches.append(('none', 'none', '////'))
    labels.append('bars used for fitting')

    handles = [ *handles_colors,]
    hmap = dict(zip(handles, [BicolorHandler(*color) for color in colors_and_hatches] ))
    figs_axes[0][0].legend(handles, labels, handler_map = hmap, loc = 'upper right', bbox_to_anchor = (0.98, 1.02), fontsize = 'xx-small', labelspacing = 0.75, frameon=False)

    chi_sq = chi_squared(theory, experiment = experiment, std = std)
    chi_sq_without_22_cold = chi_squared(theory.flatten()[:-1], experiment = experiment.flatten()[:-1], std = std.flatten()[:-1])
    chi_sq_without_22_and_20_cold = chi_squared(theory.flatten()[[0,1,2,3,4,5,6,8]], experiment = experiment.flatten()[[0,1,2,3,4,5,6,8]], std = std.flatten()[[0,1,2,3,4,5,6,8]])
    log_str = f'''For T = {plot_temperature:.2e} K, the chi-squared for ab-initio singlet potential (Phi_s = {singlet_phase:.4f} pi) and DeltaPhi = {triplet_phase-singlet_phase:.2f} calculated from the barplot for 88Sr+ is {chi_sq:.3f}.\nExcluding the measurement of the cold spin change for |2,2>|up> state, chi-squared is {chi_sq_without_22_cold:.3f}.\nExcluding the cold spin change for |2,0>|up> state as well, chi-squared is {chi_sq_without_22_and_20_cold:.3f}'''
    with open(log_path, 'w') as log_file:
        log_file.write(log_str)

    figs[1], _ax, _reduced_masses, _theory = plotPeffAverageVsMassToFig(figs[1], singlet_phase, triplet_phase, so_scaling, reduced_masses, energy_tuple_vs_mass_even, energy_tuple_vs_mass_odd, temperatures, plot_temperature, even_input_dir_name = vs_mass_even_input_dir_name, odd_input_dir_name = vs_mass_odd_input_dir_name,)
    figs_axes[1].append(_ax)
    # figs_axes[1][0].set_ylim(figs_axes[0][0].get_ylim())

    np.savetxt(data_path.with_stem(data_path.stem+'_reduced_masses'), _reduced_masses, fmt = '%.4f')
    print(f'{_theory.shape=}')
    np.savetxt(data_path.with_stem(data_path.stem+'_hpf_vs_reduced_mass'), _theory, fmt = '%.4f')

    figs[2], _ax, _reduced_masses, _theory = plotP0VsMassWithPartialWavesToFig(figs[2], singlet_phase, triplet_phase, so_scaling, reduced_masses, energy_tuple_vs_mass_even, temperatures, plot_temperature, input_dir_name = vs_mass_even_input_dir_name, transfer_input_dir_name = 'RbSr+_tcpld_momentum_transfer_vs_mass',)
    figs_axes[2].append(_ax)

    np.savetxt(data_path.with_stem(data_path.stem+'_hpf_vs_L_vs_reduced_mass'), _theory, fmt = '%.4f')

    figs[0].subplots_adjust(left = 0.1, bottom = 0.15)
    figs[1].subplots_adjust(left = 0.1, bottom = 0.15)
    figs[2].subplots_adjust(left = 0.1, bottom = 0.15)

    figs_axes[0][0].text(-0.06, 1.0, f'a', fontsize = 8, family = 'sans-serif', va = 'top', ha = 'left', transform = fig.transFigure, fontweight = 'bold')
    figs_axes[1][0].text(-0.06, 2*row_height/total_height, f'b', fontsize = 8, family = 'sans-serif', va = 'top', ha = 'left', transform = fig.transFigure, fontweight = 'bold')
    figs_axes[1][0].text(-0.06, 1*row_height/total_height, f'c', fontsize = 8, family = 'sans-serif', va = 'top', ha = 'left', transform = fig.transFigure, fontweight = 'bold')

    fig.savefig(png_path, bbox_inches='tight', pad_inches = 0)
    fig.savefig(svg_path, bbox_inches='tight', pad_inches = 0, transparent = True)
    fig.savefig(pdf_path, bbox_inches='tight', pad_inches = 0, transparent = True)

    # np.savetxt(data_path.with_stem(data_path.stem+'_reduced_masses'), _reduced_masses, fmt = '%.4f')
    # print(f'{_theory.shape=}')
    # np.savetxt(data_path.with_stem(data_path.stem+'_hpf_vs_reduced_mass'), _theory, fmt = '%.4f')

    plt.close()

def plotPeffAverageVsMassToFig(fig, singlet_phase: float, triplet_phase: float, so_scaling: float, reduced_masses: np.ndarray[float], energy_tuple_vs_mass_even: tuple[float, ...], energy_tuple_vs_mass_odd: tuple[float, ...], temperatures: np.ndarray[float] = np.array([5e-4,]), plot_temperature: float = 5e-4, even_input_dir_name: str = 'RbSr+_tcpld_80mK_vs_mass', odd_input_dir_name: str = 'RbSr+_tcpld_vs_mass_odd'):
    ## (c) Effective probability of the hyperfine energy release vs reduced mass
    probabilities_dir_name = 'probabilities'
    
    curves_names = [ f'$i_\\mathrm{{ion}} = 0$', f'$i_\\mathrm{{ion}} = \\frac{{9}}{{2}}$' ]

    abbreviations_efficiency_even = {'p0_hpf': 1.00,}
    F_in_even = 2*2
    abbreviations_efficiency_odd  = {'p0_hpf': 1.00, 'p0_hpf_exchange': 0.60}
    F_in_odd = 2*5

    nenergies = len(energy_tuple_vs_mass_even)
    E_min = min(energy_tuple_vs_mass_even)
    E_max = max(energy_tuple_vs_mass_even)

    array_paths_even = { abbreviation:  [arrays_dir_path / even_input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{triplet_phase:.4f}' / f'{so_scaling:.4f}' / f'{reduced_mass:.4f}_amu' / probabilities_dir_name / f'{abbreviation}.txt' for reduced_mass in reduced_masses] for abbreviation in abbreviations_efficiency_even.keys() }
    [ [print({abbreviation: array_path}) for array_path in array_paths if (array_path is not None and not array_path.is_file())] for abbreviation, array_paths in array_paths_even.items() ]
    p0_arrays_even = { abbreviation: np.array([np.loadtxt(array_path).reshape(len(temperatures), F_in_even+1) if (array_path is not None and array_path.is_file()) else np.full((len(temperatures), F_in_even+1), np.nan) for array_path in array_paths]) for abbreviation, array_paths in array_paths_even.items() }
    peff_arrays_even = effective_probability(np.sum([array for array in p0_arrays_even.values()], axis = 0), pmf_array = pmf_array)
    peff_arrays_even = peff_arrays_even * np.sum( [ p0_arrays_even[abbreviation]*abbreviations_efficiency_even[abbreviation] for abbreviation in abbreviations_efficiency_even.keys() ], axis = 0 ) / np.sum( [ p0_arrays_even[abbreviation] for abbreviation in abbreviations_efficiency_even.keys() ], axis = 0 )    
    peff_arrays_even = np.mean( peff_arrays_even, axis = -1 )

    nenergies = len(energy_tuple_vs_mass_odd)
    E_min = min(energy_tuple_vs_mass_odd)
    E_max = max(energy_tuple_vs_mass_odd)

    ### array_paths_even: dict with keys, values = file_name (defining p0/peff & relaxation/excit.exch.), full_path
    array_paths_odd = { abbreviation:  [arrays_dir_path / odd_input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{triplet_phase:.4f}' / f'{so_scaling:.4f}' / f'{reduced_mass:.4f}_amu' / probabilities_dir_name / f'{abbreviation}.txt' for reduced_mass in reduced_masses] for abbreviation in abbreviations_efficiency_odd.keys() }
    [ [print({abbreviation: array_path}) for array_path in array_paths if (array_path is not None and not array_path.is_file())] for abbreviation, array_paths in array_paths_odd.items() ]
    ### p0_arrays_even: dict with keys, values = file_name (defining p0/peff & relaxation/excit.exch.), loaded_array
    p0_arrays_odd = { abbreviation: np.array([np.loadtxt(array_path).reshape(len(temperatures), F_in_even+1, F_in_odd+1) if (array_path is not None and array_path.is_file()) else np.full((len(temperatures), F_in_even+1, F_in_odd+1), np.nan) for array_path in array_paths]) for abbreviation, array_paths in array_paths_odd.items() }
    # print(f'{p0_arrays_odd["p0_hpf"][0]=}')
    
    ### firstly, we add the relaxation and the excitation exchange to get the total (effective) probability of any of the two strongly exothermic outcomes
    peff_arrays_odd = effective_probability(np.sum([array for array in p0_arrays_odd.values()], axis = 0), pmf_array = pmf_array)
    ### then, we correct for the lower efficiency of measurement for the hyperfine excitation exchange
    peff_arrays_odd = peff_arrays_odd * np.sum( [ p0_arrays_odd[abbreviation]*abbreviations_efficiency_odd[abbreviation] for abbreviation in abbreviations_efficiency_odd.keys() ], axis = 0 ) / np.sum( [ p0_arrays_odd[abbreviation] for abbreviation in abbreviations_efficiency_odd.keys() ], axis = 0 )
    peff_arrays_odd = np.mean( peff_arrays_odd, axis = (-2, -1) )


    exp_hpf_isotopes = np.loadtxt(data_dir_path / 'exp_data' / 'isotopes_hpf.dat')
    reduced_masses_experimental = np.array([red_mass_87Rb_84Sr_amu, red_mass_87Rb_86Sr_amu, red_mass_87Rb_87Sr_amu, red_mass_87Rb_88Sr_amu])
    reduced_masses_labels = [ f'$\\mathrm{{{{}}^{{84}}Sr^+}}$', f'$\\mathrm{{{{}}^{{86}}Sr^+}}$', f'$\\mathrm{{{{}}^{{87}}Sr^+}}$', f'$\\mathrm{{{{}}^{{88}}Sr^+}}$' ]
    peff_experiment = exp_hpf_isotopes[0,:]
    peff_std_experiment = exp_hpf_isotopes[1,:]
    #dpeff = 1e-3
    #p0_std = (p0(peff_experiment+dpeff/2, pmf_array=pmf_array)-p0(peff_experiment-dpeff/2, pmf_array=pmf_array))/dpeff * peff_std_experiment
    experiment = peff_experiment# if enhanced else p0(peff_experiment, pmf_array)
    std = peff_std_experiment# if enhanced else p0_std

    T_index = np.nonzero(temperatures == plot_temperature)[0][0]
    theory = np.moveaxis( [peff_arrays_even[:,T_index], peff_arrays_odd[:,T_index]], 0, -1)
    # print(f'{theory=}')

    even_color = 'firebrick'
    odd_color = 'darkmagenta'
    theory_formattings = [ {'color': even_color, 'linewidth': 1.25},
                          {'color': odd_color, 'linewidth': 1.25}
                          ]
    # theory_distinguished_formattings = [ {'color': 'k', 'linewidth': 4, 'linestyle':  (1.05,(0.1,2)), 'dash_capstyle': 'round' } for exp in experiment]
    # experiment_formattings = [ {'color': 'firebrick', 'dash_capstyle': 'round', } for exp in experiment]

    fig_ax = fig.add_subplot()
    # print(f'{reduced_masses_experimental=}, {experiment=}')
    fig_ax.scatter(reduced_masses_experimental[[0,1,3]], experiment[[0,1,3]], s = 16, c = 'firebrick', marker = 'x', edgecolors =  'firebrick', linewidths = None)
    fig_ax.errorbar(reduced_masses_experimental[[0,1,3]], experiment[[0,1,3]], std[[0,1,3]], ecolor = 'firebrick', capsize = 4, linestyle = 'None')
    fig_ax.scatter(reduced_masses_experimental[2], experiment[2], s = 16, c = 'darkmagenta', marker = 'x', edgecolors =  'darkmagenta', linewidths = None)
    fig_ax.errorbar(reduced_masses_experimental[2], experiment[2], std[2], ecolor = 'darkmagenta', capsize = 4, linestyle = 'None')
    fig_ax = ValuesVsModelParameters.plotValuestoAxis(fig_ax, reduced_masses, theory, experiment=None, std=None, theory_distinguished=None, theory_formattings = theory_formattings, theory_distinguished_formattings=None)
    fig_ax.set_ylim(0,0.5)# 1.2*fig_ax.get_ylim()[1])
    PhaseTicks.linearStr(fig_ax.yaxis, 0.1, 0.05, '${x:.1f}$')
    PhaseTicks.linearStr(fig_ax.xaxis, 0.5, 0.1, '${x:.1f}$')
    # fig_ax.set_xticks([ 43.0, 43.5, *reduced_masses_experimental], labels = [ '$43.0$', '$43.5$', f'${{}}^{{84}}\\mathrm{{Sr^+}}$', f'${{}}^{{86}}\\mathrm{{Sr^+}}$', f'${{}}^{{87}}\\mathrm{{Sr^+}}$', f'${{}}^{{88}}\\mathrm{{Sr^+}}$' ])
    xx_text, yy_text, alignments = reduced_masses_experimental, experiment, [ {'ha': 'center', 'va': 'top'}, {'ha': 'center', 'va': 'top'}, {'ha': 'right', 'va': 'baseline'}, {'ha': 'right', 'va': 'baseline'} ]
    xx_text[[3,]] += 0.01
    yy_text[[0,1]] += -0.045
    yy_text[[2,3]] += 0.03
    for i in [0, 1, 3]:
        fig_ax.text(xx_text[i], yy_text[i], reduced_masses_labels[i], color = 'firebrick', fontsize = 'x-small', **alignments[i])
    fig_ax.text(xx_text[2], yy_text[2], reduced_masses_labels[2], color = 'darkmagenta', fontsize = 'x-small', **alignments[i])

    for i, curve_name in enumerate(curves_names):
        # [print(f'{i}, {line.get_xydata()=}') for i, line in enumerate(fig_ax.get_lines())]
        if len(fig_ax.get_lines()[i+2*3].get_xydata()) > 0: fig_ax.get_lines()[i+2*3].set_label(curve_name) 
    fig_ax.legend(frameon = False, loc = 'upper right', bbox_to_anchor = (.97, .98), fontsize = 'x-small', labelspacing = .1)
    # xvals = (fig_ax.get_xlim()[0] + 0.1*(fig_ax.get_xlim()[1]-fig_ax.get_xlim()[0]), fig_ax.get_xlim()[0] + 0.6*(fig_ax.get_xlim()[1]-fig_ax.get_xlim()[0]))
    # labelLines(fig_ax.get_lines(), xvals = xvals, align = False, outline_width=2, color = 'white', fontsize = matplotlib.rcParams["xtick.labelsize"], )
    # labelLines(fig_ax.get_lines(), xvals = xvals, align = False, outline_color = None, yoffsets= -0*6.7e-3*(fig_ax.get_ylim()[1]-fig_ax.get_ylim()[0]), fontsize = matplotlib.rcParams["xtick.labelsize"], )
    # props = dict(boxstyle='round', facecolor='none', edgecolor='midnightblue')
    # fig_ax.text(0.03, 0.10, f'$\\Delta\\Phi_\\mathrm{{fit}} = {(magnetic_phases[0][1]-magnetic_phases[0][0])%1:.2f}\\pi$', va = 'center', ha = 'left', transform = fig_ax.transAxes, bbox = props)

    ylabel = f'$p_\\mathrm{{eff}}^\\mathrm{{hpf}}$ (state-averaged)'# if enhanced else f'$p_0$'
    fig_ax.set_ylabel(ylabel)

    fig_ax.set_xlabel(f'reduced mass (a.m.u.)')

    return fig, fig_ax, reduced_masses, theory

def plotP0VsMassWithPartialWavesToFig(fig, singlet_phase: float, triplet_phase: float, so_scaling: float, reduced_masses: np.ndarray[float], energy_tuple_vs_mass: tuple[float, ...], temperatures: np.ndarray[float] = np.array([5e-4,]), plot_temperature: float = 5e-4, input_dir_name: str = 'RbSr+_tcpld_vs_mass', transfer_input_dir_name: str = 'RbSr+_tcpld_momentum_transfer_vs_mass',):
    ## (c) Probability of the hyperfine energy release (only spin exchange?) vs reduced mass

    nenergies = len(energy_tuple_vs_mass)
    E_min = min(energy_tuple_vs_mass)
    E_max = max(energy_tuple_vs_mass)
    # print(f'{energy_tuple_vs_mass = }\n{nenergies = }\n{E_min = }\n{E_max = }')

    pickle_path = pickles_dir_path / input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{triplet_phase:.4f}' / f'{so_scaling:.4f}' / f'{reduced_masses[0]:.4f}_amu.pickle'
    transfer_pickle_path = pickles_dir_path / transfer_input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{triplet_phase:.4f}' / f'{0.0:.4f}' / f'{reduced_masses[0]:.4f}_amu.pickle'

    s_matrix_collection = SMatrixCollection.fromPickle(pickle_path)
    l_max = int(max(key[0].L for s_matrix in s_matrix_collection.matrixCollection.values() for key in s_matrix.matrix.keys())/2)

    transfer_s_matrix_collection = SMatrixCollection.fromPickle(transfer_pickle_path)
    transfer_l_max = int(max(key[0].L for s_matrix in transfer_s_matrix_collection.matrixCollection.values() for key in s_matrix.matrix.keys())/2)    

    so_scaling = s_matrix_collection.spinOrbitParameter[0]
    reduced_mass_amu = s_matrix_collection.reducedMass[0]/amu_to_au

    abbreviation_k_L = 'hpf'
    F_in_even = 2*2

    F_in, MF_in, S_in, MS_in = F_in_even, 0, 1, 1
    F_out, MF_out, S_out, MS_out = 2, MF_in+2, 1, MS_in-2
    print(f'{singlet_phase = }, {triplet_phase = }, {so_scaling = }, {reduced_masses = }')
    k_archive_paths = [arrays_dir_path / input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{triplet_phase:.4f}' / f'{so_scaling:.4f}' / f'{reduced_mass:.4f}_amu.zip' for reduced_mass in reduced_masses]
    k_L_E_array_paths = [arrays_dir_path / input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{triplet_phase:.4f}' / f'{so_scaling:.4f}' / f'{reduced_mass:.4f}_amu' / f'k_L_E' / f'{abbreviation_k_L}' / f'OUT_{F_out}_{MF_out}_{S_out}_{MS_out}_IN_{F_in}_{MF_in}_{S_in}_{MS_in}.txt' for reduced_mass in reduced_masses]
    k_m_L_E_array_paths = [arrays_dir_path / input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{triplet_phase:.4f}' / f'{so_scaling:.4f}' / f'{reduced_mass:.4f}_amu' / f'k_m_L_E' / f'{abbreviation_k_L}' / f'OUT_{F_out}_{MF_out}_{S_out}_{MS_out}_IN_{F_in}_{MF_in}_{S_in}_{MS_in}.txt' for reduced_mass in reduced_masses]

    # print("Starting extracting!")
    # for archive_path in k_archive_paths:
    #     print(archive_path)
    #     with zipfile.ZipFile(archive_path, 'r') as zObject:
    #         if not (archive_path.with_suffix('') / Path(f'k_L_E/{abbreviation_k_L}/OUT_{F_out}_{MF_out}_{S_out}_{MS_out}_IN_{F_in}_{MF_in}_{S_in}_{MS_in}.txt')).is_file():
    #             zObject.extract(f'k_L_E/{abbreviation_k_L}/OUT_{F_out}_{MF_out}_{S_out}_{MS_out}_IN_{F_in}_{MF_in}_{S_in}_{MS_in}.txt', archive_path.with_suffix(''))
    #         if not (archive_path.with_suffix('') / Path(f'k_m_L_E/{abbreviation_k_L}/OUT_{F_out}_{MF_out}_{S_out}_{MS_out}_IN_{F_in}_{MF_in}_{S_in}_{MS_in}.txt')).is_file():
    #             zObject.extract(f'k_m_L_E/{abbreviation_k_L}/OUT_{F_out}_{MF_out}_{S_out}_{MS_out}_IN_{F_in}_{MF_in}_{S_in}_{MS_in}.txt', archive_path.with_suffix(''))
    # print("Finished extracting!")

    #### here we get arrays with indices (reduced_mass_index, L_index, E_index)
    print("Starting loading k_L_E arrays")
    k_L_E_arrays = np.array([ np.loadtxt(array_path) if (array_path is not None and array_path.is_file()) else np.full((l_max+1,50), np.nan) for array_path in k_L_E_array_paths ])
    print("Starting loading k_m_L_E arrays")
    k_m_L_E_arrays = np.array([ np.loadtxt(array_path) if (array_path is not None and array_path.is_file()) else np.full((transfer_l_max,50), np.nan) for array_path in k_m_L_E_array_paths ])
    print("Finished loading the arrays")

    distribution_arrays = [np.fromiter(n_root_iterator(temperature = temperature, E_min = min(s_matrix_collection.collisionEnergy), E_max = max(s_matrix_collection.collisionEnergy), N = len(s_matrix_collection.collisionEnergy), n = 3), dtype = float) for temperature in temperatures]
    transfer_distribution_arrays = [np.fromiter(n_root_iterator(temperature = temperature, E_min = min(transfer_s_matrix_collection.collisionEnergy), E_max = max(transfer_s_matrix_collection.collisionEnergy), N = len(transfer_s_matrix_collection.collisionEnergy), n = 3), dtype = float) for temperature in temperatures]
    
    print("Starting calculating energy averages")
    average_rate_arrays = np.array( [s_matrix_collection.thermalAverage(k_L_E_arrays, distribution_array) for distribution_array in distribution_arrays ] )
    average_rate_arrays = np.moveaxis(average_rate_arrays, -1, 0)
    average_momentum_transfer_arrays = np.array( [ transfer_s_matrix_collection.thermalAverage(k_m_L_E_arrays.sum(axis=1), transfer_distribution_array) for transfer_distribution_array in transfer_distribution_arrays ] )
    print("Finished calculating energy averages")
    print(f'{average_rate_arrays.shape = }, {average_momentum_transfer_arrays.shape = }')
    probability_arrays = average_rate_arrays / average_momentum_transfer_arrays
    probability_arrays = np.moveaxis(probability_arrays,0,-1)
    print(f'{probability_arrays.shape}')
    # probability_arrays = probability_arrays.squeeze()

    probability_vs_L_vs_mass_path = k_archive_paths[0].parent / 'p0_vs_L_vs_reduced_mass'

    temperatures_str = np.array2string( np.array(temperatures),formatter={'float_kind':lambda x: '%.2e' % x} )
    # momentum_transfer_str = np.array2string(average_momentum_transfer_arrays.reshape(average_momentum_transfer_arrays.shape[0], -1)[:,0], formatter={'float_kind':lambda x: '%.4e' % x} )

    print("------------------------------------------------------------------------")
    print(f'The bare probabilities p_0 of the hyperfine relaxation for phases = ({singlet_phase:.4f}, {triplet_phase:.4f}), {so_scaling = }, {reduced_mass_amu = } a.m.u., temperatures: {temperatures_str} K, the maximum L for the momentum-transfer rates calculations: {transfer_l_max} are:')
    print(probability_arrays, '\n')

    np.savetxt(probability_vs_L_vs_mass_path, probability_arrays.reshape(probability_arrays.shape[0],-1), fmt = '%.10f', header = f'[Original shape: {probability_arrays.shape}]\nThe bare probabilities of the hyperfine relaxation exchange; indices: (L, reduced_mass).\nThe values of reduced mass: {np.array(s_matrix_collection.reducedMass)/amu_to_au} a.m.u.\nThe singlet, triplet semiclassical phases: ({singlet_phase:.4f}, {triplet_phase:.4f}). The scaling of the short-range part of lambda_SO: {so_scaling}.\nThe maximum L: {l_max}. The maximum change of L: +/-4.\nTemperatures: {temperatures_str} K.\nThe maximum L for the momentum-transfer rates calculations: {transfer_l_max}.')

    T_index = np.nonzero(temperatures == plot_temperature)[0][0]
    theory = probability_arrays[T_index,:,:]

    even_color = 'firebrick'
    odd_color = 'darkmagenta'
    color_map = matplotlib.colormaps['viridis']
    norm = matplotlib.colors.Normalize(vmin=0, vmax=l_max, clip = False)
    # theory_colors = [color_map(norm(L)) for L in range(l_max+1)]
    theory_formattings = [ #{'color': even_color, 'linewidth': 1.25},
                          *[{'color': color_map(norm(L)), 'linewidth': 0.25} for L in range(0,l_max+1)]
                        #   {'color': odd_color, 'linewidth': 1.25}
                          ]

    fig_ax = fig.add_subplot()
    fig_ax = ValuesVsModelParameters.plotValuestoAxis(fig_ax, reduced_masses, theory, experiment=None, std=None, theory_distinguished=None, theory_formattings = theory_formattings, theory_distinguished_formattings=None)
    fig_ax.set_ylim(0,1.2*np.amax(theory))# 1.2*fig_ax.get_ylim()[1])
    PhaseTicks.linearStr(fig_ax.yaxis, 0.01, 0.005, '${x:.1f}$')
    PhaseTicks.linearStr(fig_ax.xaxis, 0.5, 0.1, '${x:.1f}$')
    # fig_ax.legend(frameon = False, loc = 'upper right', bbox_to_anchor = (.97, .98), fontsize = 'x-small', labelspacing = .1)
    ylabel = f'$p_0$'# if enhanced else f'$p_0$'
    fig_ax.set_ylabel(ylabel)
    fig_ax.set_xlabel(f'reduced mass (a.m.u.)')

    # theory_distinguished_formattings = [ {'color': 'k', 'linewidth': 4, 'linestyle':  (1.05,(0.1,2)), 'dash_capstyle': 'round' } for exp in experiment]
    # experiment_formattings = [ {'color': 'firebrick', 'dash_capstyle': 'round', } for exp in experiment]

    return fig, fig_ax, reduced_masses, theory


def main():
    parser_description = "This is a python script for running molscat, collecting and pickling S-matrices, and calculating effective probabilities."
    parser = argparse.ArgumentParser(description=parser_description)
    parser.add_argument("-s", "--singlet_phase", type = float, default = None, help = "The singlet semiclassical phase modulo pi in multiples of pi.")
    parser.add_argument("-t", "--triplet_phase", type = float, default = None, help = "The triplet semiclassical phase modulo pi in multiples of pi.")
    parser.add_argument("-d", "--phase_difference", type = float, default = None, help = "The singlet-triplet semiclassical phase difference modulo pi in multiples of pi.")
    parser.add_argument("--so_scaling", type = float, default = 0.320, help = "Value of the SO scaling.")
    parser.add_argument("--mass_min", type = float, default = 42.48, help = "Minimum reduced mass for the grid (in a.m.u.).")
    parser.add_argument("--mass_max", type = float, default = 43.8, help = "Maximum reduced mass for the grid (in a.m.u.).")
    parser.add_argument("--dmass", type = float, default = 0.01, help = "Mass step (in a.m.u.).")

    parser.add_argument("--nenergies_barplot", type = int, default = 50, help = "Number of energy values in a grid.")
    parser.add_argument("--E_min_barplot", type = float, default = 4e-7, help = "Lowest energy value in the grid.")
    parser.add_argument("--E_max_barplot", type = float, default = 4e-3, help = "Highest energy value in the grid.")

    parser.add_argument("--nenergies_even", type = int, default = 50, help = "Number of energy values in a grid.")
    parser.add_argument("--E_min_even", type = float, default = 4e-7, help = "Lowest energy value in the grid.")
    parser.add_argument("--E_max_even", type = float, default = 4e-3, help = "Highest energy value in the grid.")

    parser.add_argument("--nenergies_odd", type = int, default = 50, help = "Number of energy values in a grid.")
    parser.add_argument("--E_min_odd", type = float, default = 4e-7, help = "Lowest energy value in the grid.")
    parser.add_argument("--E_max_odd", type = float, default = 4e-3, help = "Highest energy value in the grid.")
    parser.add_argument("--n_grid", type = int, default = 3, help = "n parameter for the nth-root energy grid.")
    
    parser.add_argument("-T", "--temperatures", nargs='*', type = float, default = None, help = "Temperature in the Maxwell-Boltzmann distributions (in kelvins).")
    parser.add_argument("--nT", type = int, default = 10, help = "Number of temperatures included in the calculations.")
    parser.add_argument("--logT_min", type = float, default = -4)
    parser.add_argument("--logT_max", type = float, default = -3)

    parser.add_argument("--barplot_input_dir_name", type = str, default = 'RbSr+_fmf_so_scaling', help = "Name of the directory with the molscat inputs")
    parser.add_argument("--barplot_SE_input_dir_name", type = str, default = 'RbSr+_fmf_vs_DPhi_SE', help = "Name of the directory with the molscat inputs")
    parser.add_argument("--vs_mass_even_input_dir_name", type = str, default = 'RbSr+_tcpld_vs_mass', help = "Name of the directory with the molscat inputs")
    parser.add_argument("--vs_mass_odd_input_dir_name", type = str, default = 'RbSr+_tcpld_vs_mass_odd', help = "Name of the directory with the molscat inputs")

    parser.add_argument("--fmf_barplot", action = 'store_true', help = "Assume that the scattering calculations in molscat were done with the fmf basis set. Change directory structure for arrays.")

    parser.add_argument("--journal", type = str, default = 'NatCommun', help = "Name of the journal to prepare the plots for.")
    parser.add_argument("--plot_p0", action = 'store_true', help = "If included, the short-range probability p0 will be plotted instead of peff.")
    args = parser.parse_args()

    n = args.n_grid
    energy_tuple_barplot = tuple( round(n_root_scale(i, args.E_min_barplot, args.E_max_barplot, args.nenergies_barplot-1, n = n), sigfigs = 11) for i in range(args.nenergies_barplot) )
    energy_tuple_vs_mass_even = tuple( round(n_root_scale(i, args.E_min_even, args.E_max_even, args.nenergies_even-1, n = n), sigfigs = 11) for i in range(args.nenergies_even) )
    energy_tuple_vs_mass_odd = tuple( round(n_root_scale(i, args.E_min_odd, args.E_max_odd, args.nenergies_odd-1, n = n), sigfigs = 11) for i in range(args.nenergies_odd) )

    args.singlet_phase = args.singlet_phase if args.singlet_phase is not None else default_singlet_phase_function(1.0)
    args.triplet_phase = args.triplet_phase if args.triplet_phase is not None else args.singlet_phase + args.phase_difference if args.phase_difference is not None else default_triplet_phase_function(1.0)

    phases = ((args.singlet_phase, args.triplet_phase),)
    singlet_phase = args.singlet_phase
    triplet_phase = args.triplet_phase

    so_scaling_value = args.so_scaling

    # reduced_masses = np.linspace(args.mass_min, args.mass_max, args.nmasses)
    reduced_masses = np.arange(args.mass_min, args.mass_max+0.5*args.dmass, args.dmass)

    if args.temperatures is None:
        temperatures = list(np.logspace(args.logT_min, args.logT_max, args.nT))
        temperatures.append(5e-4)
        temperatures = np.array(sorted(temperatures))
    else:
        temperatures = np.array(args.temperatures)

    print(f'{red_mass_87Rb_84Sr_amu=}')
    print(f'{red_mass_87Rb_86Sr_amu=}')
    print(f'{red_mass_87Rb_87Sr_amu=}')
    print(f'{red_mass_87Rb_88Sr_amu=}')

    [plotFig2(singlet_phase, triplet_phase, so_scaling_value, reduced_masses, energy_tuple_barplot= energy_tuple_barplot, energy_tuple_vs_mass_even = energy_tuple_vs_mass_even, energy_tuple_vs_mass_odd = energy_tuple_vs_mass_odd, temperatures = temperatures, plot_temperature = temperature, barplot_input_dir_name = args.barplot_input_dir_name, barplot_SE_input_dir_name = args.barplot_SE_input_dir_name, vs_mass_even_input_dir_name = args.vs_mass_even_input_dir_name, vs_mass_odd_input_dir_name = args.vs_mass_odd_input_dir_name, fmf_barplot = args.fmf_barplot, journal_name = args.journal, plot_p0 = args.plot_p0) for temperature in temperatures]

if __name__ == '__main__':
    main()