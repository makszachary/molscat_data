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
from _molscat_data.visualize import ContourMap, ValuesVsModelParameters, PhaseTicks, Barplot, BicolorHandler
from _molscat_data.physical_constants import red_mass_87Rb_84Sr_amu, red_mass_87Rb_86Sr_amu, red_mass_87Rb_87Sr_amu, red_mass_87Rb_88Sr_amu

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

def plotFig2(singlet_phase: float, triplet_phase: float, so_scaling: float, reduced_masses: np.ndarray[float], energy_tuple_barplot: tuple[float, ...], energy_tuple_vs_mass_even: tuple[float, ...], energy_tuple_vs_mass_odd: tuple[float, ...], temperatures: np.ndarray[float] = np.array([5e-4,]), plot_temperature: float = 5e-4, barplot_input_dir_name: str = 'RbSr+_tcpld_80mK_0.01_step', barplot_SE_input_dir_name: str = 'RbSr+_tcpld_80mK_0.01_step_SE', vs_mass_even_input_dir_name = 'RbSr+_tcpld_80mK_vs_mass', vs_mass_odd_input_dir_name = 'RbSr+_tcpld_vs_mass_odd', journal_name = 'NatCommun'):
    plt.style.use(Path(__file__).parent / 'mpl_style_sheets' / f'{journal_name}.mplstyle')
    # nenergies = len(energy_tuple_barplot)
    # E_min = min(energy_tuple_barplot)
    # E_max = max(energy_tuple_barplot)
    png_path = plots_dir_path / 'paper' / f'{journal_name}' / 'Fig2' / f'Fig2_{plot_temperature:.2e}K.png'
    pdf_path = png_path.with_suffix('.pdf')
    svg_path = png_path.with_suffix('.svg')
    png_path.parent.mkdir(parents = True, exist_ok = True)
    
    
    cm = 1/2.54
    ws, hs = 0.05, 0.05
    nrows = 4
    row_height = 3
    vpad = 1
    total_height = nrows*row_height + (nrows-1)*vpad
    figsize = (8.8*cm, total_height*cm)
    dpi = 1200
    fig = plt.figure(figsize = figsize, dpi = dpi)
    gs_Figure = gridspec.GridSpec(nrows, 1, fig, hspace = hs, wspace = ws, height_ratios = [1 for row in range(nrows)])
    # figs = fig.subfigures(2, 2, wspace = ws, hspace = hs)
    fig0 = fig.add_subfigure(gs_Figure[0])
    fig1 = fig.add_subfigure(gs_Figure[1])
    fig2 = fig.add_subfigure(gs_Figure[2])
    fig3 = fig.add_subfigure(gs_Figure[3])
    # fig3 = fig.add_subfigure(gs_Figure[2,1])

    
    nenergies = len(energy_tuple_barplot)
    E_min = min(energy_tuple_barplot)
    E_max = max(energy_tuple_barplot)
    probabilities_dir_name = 'probabilities'

    arrays_path_hpf = arrays_dir_path / barplot_input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{triplet_phase:.4f}' / f'{so_scaling:.4f}' / probabilities_dir_name / 'hpf.txt'
    SE_arrays_path_hpf = arrays_dir_path / barplot_SE_input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{triplet_phase:.4f}' / probabilities_dir_name / 'hpf.txt'
    arrays_path_cold_higher = arrays_dir_path / barplot_input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{triplet_phase:.4f}' / f'{so_scaling:.4f}' / probabilities_dir_name / 'cold_higher.txt'
    SE_arrays_path_cold_higher = arrays_dir_path / barplot_SE_input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{triplet_phase:.4f}' / probabilities_dir_name / 'cold_higher.txt'

    arrays_hpf = np.loadtxt(arrays_path_hpf)
    SE_arrays_hpf = np.loadtxt(SE_arrays_path_hpf)
    arrays_cold_higher = np.loadtxt(arrays_path_cold_higher)
    SE_arrays_cold_higher = np.loadtxt(SE_arrays_path_cold_higher)

    T_index = np.nonzero(temperatures == plot_temperature)[0][0]
    theory_hpf= arrays_hpf[T_index,:]
    theory_SE_hpf = SE_arrays_hpf[T_index,:]
    theory_cold_higher = arrays_cold_higher[T_index,:]
    theory_SE_cold_higher = SE_arrays_cold_higher[T_index,:]

    experiment_std_path_hpf = Path(__file__).parents[1] / 'data' / 'exp_data' / 'single_ion_hpf.dat'
    experiment_std_path_cold_higher = Path(__file__).parents[1] / 'data' / 'exp_data' / 'single_ion_cold_higher.dat'
    experiment_hpf = np.loadtxt(experiment_std_path_hpf)[0]
    std_hpf = np.loadtxt(experiment_std_path_hpf)[1]
    experiment_cold_higher = np.loadtxt(experiment_std_path_cold_higher)[0]
    std_cold_higher = np.loadtxt(experiment_std_path_cold_higher)[1]
    

    f_max = 2
    barplot_labels = [ '$\\left|\\right.$'+str(int(f_max))+', '+str(int(mf))+'$\\left.\\right>$' for mf in np.arange (-f_max, f_max+1)]

    fig0_ax = fig0.add_subplot()
    bars_formatting_hpf = { 'facecolor': 'indianred', 'edgecolor': 'black', 'alpha': 0.9, 'ecolor': 'black', 'capsize': 3 }
    exp_bars_formatting_hpf = { 'facecolor': 'firebrick', 'edgecolor': 'black', 'alpha': 0.9, 'ecolor': 'black', 'capsize': 3 }
    SE_bars_formatting_hpf = { 'facecolor': 'firebrick', 'edgecolor': 'black', 'alpha': 0.9, 'ecolor': 'black', 'capsize': 3 }
    bars_formatting_cold_higher = { 'facecolor': 'royalblue', 'edgecolor': 'black', 'alpha': 0.9, 'ecolor': 'black', 'capsize': 3 }
    exp_bars_formatting_cold_higher = { 'facecolor': 'midnightblue', 'edgecolor': 'black', 'alpha': 0.9, 'ecolor': 'black', 'capsize': 3 }
    SE_bars_formatting_cold_higher = { 'facecolor': 'midnightblue', 'edgecolor': 'black', 'alpha': 0.9, 'ecolor': 'black', 'capsize': 3 }

    # fig0_ax = Barplot.plotBarplotToAxes(fig0_ax, theory_hpf, experiment_hpf, std_hpf, barplot_labels, theory_SE_hpf, bars_formatting = bars_formatting_hpf, exp_bars_formatting = exp_bars_formatting_hpf, SE_bars_formatting = SE_bars_formatting_hpf)
    theory = np.array([ theory_hpf, theory_cold_higher ])
    theory_SE = np.array([ theory_SE_hpf, theory_SE_cold_higher ])
    experiment = np.array([ experiment_hpf, experiment_cold_higher ])
    std = np.array([ std_hpf, std_cold_higher ])
    bars_formatting = [ bars_formatting_hpf, bars_formatting_cold_higher ]
    SE_bars_formatting = [ SE_bars_formatting_hpf, SE_bars_formatting_cold_higher ]
    fig0_ax = Barplot.plotBarplotConciseToAxes(fig0_ax, theory, experiment, std, barplot_labels, theory_SE, bars_formatting = bars_formatting, SE_bars_formatting = SE_bars_formatting)
    PhaseTicks.linearStr(fig0_ax.yaxis, 0.1, 0.05, '${x:.1f}$')
    fig0_ax.set_ylim(0, 0.61)

    ylabel = f'$p_\mathrm{{eff}}$'# if enhanced else f'$p_0$'
    fig0_ax.set_ylabel(ylabel)

    labels_and_colors = { 'hyperfine relaxation\n(calculated w/o & with SO coupling)': SE_bars_formatting_hpf['facecolor'], }
    handles_colors = [ plt.Rectangle((0,0), 1, 1, facecolor = labels_and_colors[color_label], edgecolor = 'k', hatch = '' ) for color_label in labels_and_colors.keys() ]
    colors_and_hatches = [ *[ (SE_bars_formatting_hpf['facecolor'], bars_formatting_hpf['facecolor'], ''), ],]
    labels = [ *list(labels_and_colors.keys()),]
    handles = [ *handles_colors,]
    hmap = dict(zip(handles, [BicolorHandler(*color) for color in colors_and_hatches] ))
    fig0_ax.legend(handles, labels, handler_map = hmap, loc = 'upper right', bbox_to_anchor = (1, 1), fontsize = 'xx-small', labelspacing = 1)


    fig1_ax = fig1.add_subplot()
    bars_formatting_cold_higher = { 'facecolor': 'royalblue', 'edgecolor': 'black', 'alpha': 0.9, 'ecolor': 'black', 'capsize': 3 }
    exp_bars_formatting_cold_higher = { 'facecolor': 'midnightblue', 'edgecolor': 'black', 'alpha': 0.9, 'ecolor': 'black', 'capsize': 3 }
    SE_bars_formatting_cold_higher = { 'facecolor': 'midnightblue', 'edgecolor': 'black', 'alpha': 0.9, 'ecolor': 'black', 'capsize': 3 }

    fig1_ax = Barplot.plotBarplotToAxes(fig1_ax, theory_cold_higher, experiment_cold_higher, std_cold_higher, barplot_labels, SE_theory = theory_SE_cold_higher, bars_formatting = bars_formatting_cold_higher, exp_bars_formatting = exp_bars_formatting_cold_higher, SE_bars_formatting = SE_bars_formatting_cold_higher)
    PhaseTicks.linearStr(fig1_ax.yaxis, 0.1, 0.05, '${x:.1f}$')
    fig1_ax.set_ylim(fig0_ax.get_ylim())

    ylabel = f'$p_\mathrm{{eff}}$'# if enhanced else f'$p_0$'
    fig1_ax.set_ylabel(ylabel)

    labels_and_colors = { 'cold spin change\n(calculated w/o & with SO coupling)': SE_bars_formatting_cold_higher['facecolor'], }    
    handles_colors = [ plt.Rectangle((0,0), 1, 1, facecolor = labels_and_colors[color_label], edgecolor = 'k', hatch = '' ) for color_label in labels_and_colors.keys() ]
    colors_and_hatches = [ *[ (SE_bars_formatting_cold_higher['facecolor'], bars_formatting_cold_higher['facecolor'], ''), ],]
    labels = [ *list(labels_and_colors.keys()),]
    handles = [ *handles_colors,]
    hmap = dict(zip(handles, [BicolorHandler(*color) for color in colors_and_hatches] ))
    fig1_ax.legend(handles, labels, handler_map = hmap, loc = 'upper right', bbox_to_anchor = (1, 1), fontsize = 'xx-small', labelspacing = 1)


    arrays_path_hpf = arrays_dir_path / barplot_input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{triplet_phase:.4f}' / f'{so_scaling:.4f}' / probabilities_dir_name / 'hpf.txt'
    arrays_path_cold_higher = arrays_dir_path / barplot_input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{triplet_phase:.4f}' / f'{so_scaling:.4f}' / probabilities_dir_name / 'cold_higher.txt'   

    arrays_hpf = np.loadtxt(arrays_path_hpf)
    arrays_cold_higher = np.loadtxt(arrays_path_cold_higher)

    fig2_ax = fig2.add_subplot()
    fig2_ax = ValuesVsModelParameters.plotValuestoAxis(fig2_ax, temperatures, arrays_hpf)
    fig2_ax.set_xscale('log')
    fig2_ax.set_xticks([1e-4, plot_temperature, 1e-3, 1e-2], labels = ['$0.1$', f'$T_\\mathrm{{exp}}$', '$1$', '$10$'])

    fig2_ax.set_ylim(0,0.61)
    PhaseTicks.linearStr(fig2_ax.yaxis, 0.2, 0.1, '${x:.1f}$')

    fig2_ax.set_xlabel(f'$T\\,(\\mathrm{{mK}})$')
    fig2_ax.set_ylabel(f'$p_\mathrm{{eff}}^\mathrm{{hpf}}$')


    f_max = 2
    lines_labels = [ '$\\left|\\right.$'+str(int(f_max))+', '+str(int(mf))+'$\\left.\\right>$' for mf in np.arange (-f_max, f_max+1)]
    for i, lbl in enumerate(lines_labels):
        fig2_ax.get_lines()[i].set_label(lbl) 
    xvals = (fig2_ax.get_xlim()[0] + 1e-3*(fig2_ax.get_xlim()[1]-fig2_ax.get_xlim()[0]), fig2_ax.get_xlim()[0] + 0.8*(fig2_ax.get_xlim()[1]-fig2_ax.get_xlim()[0]))
    # xvals = 2e-3
    labelLines(fig2_ax.get_lines(), xvals = xvals, align = False, outline_width=4, color = 'white', va = 'bottom', fontsize = matplotlib.rcParams["xtick.labelsize"], )
    labelLines(fig2_ax.get_lines(), xvals = xvals, align = False, outline_color = None, va = 'bottom', yoffsets= -0*6.7e-3*(fig2_ax.get_ylim()[1]-fig2_ax.get_ylim()[0]), fontsize = matplotlib.rcParams["xtick.labelsize"], )

    fig3, fig3_ax = plotPeffAverageVsMassToFig(fig3, singlet_phase, triplet_phase, so_scaling, reduced_masses, energy_tuple_vs_mass_even, energy_tuple_vs_mass_odd, temperatures, plot_temperature, even_input_dir_name = vs_mass_even_input_dir_name, odd_input_dir_name = vs_mass_odd_input_dir_name)
    
    fig0.subplots_adjust(left = 0.1, bottom = 0.15)
    fig1.subplots_adjust(left = 0.1, bottom = 0.15)
    fig2.subplots_adjust(left = 0.1, bottom = 0.25)
    fig3.subplots_adjust(left = 0.1, bottom = 0.15)

    fig.savefig(png_path, bbox_inches='tight', pad_inches = 0)
    fig.savefig(svg_path, bbox_inches='tight', pad_inches = 0, transparent = True)
    fig.savefig(pdf_path, bbox_inches='tight', pad_inches = 0, transparent = True)

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

    array_paths_odd = { abbreviation:  [arrays_dir_path / odd_input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{triplet_phase:.4f}' / f'{so_scaling:.4f}' / f'{reduced_mass:.4f}_amu' / probabilities_dir_name / f'{abbreviation}.txt' for reduced_mass in reduced_masses] for abbreviation in abbreviations_efficiency_odd.keys() }
    [ [print({abbreviation: array_path}) for array_path in array_paths if (array_path is not None and not array_path.is_file())] for abbreviation, array_paths in array_paths_odd.items() ]
    p0_arrays_odd = { abbreviation: np.array([np.loadtxt(array_path).reshape(len(temperatures), F_in_even+1, F_in_odd+1) if (array_path is not None and array_path.is_file()) else np.full((len(temperatures), F_in_even+1, F_in_odd+1), np.nan) for array_path in array_paths]) for abbreviation, array_paths in array_paths_odd.items() }
    print(f'{p0_arrays_odd["p0_hpf"][0]=}')
    peff_arrays_odd = effective_probability(np.sum([array for array in p0_arrays_odd.values()], axis = 0), pmf_array = pmf_array)
    peff_arrays_odd = peff_arrays_odd * np.sum( [ p0_arrays_odd[abbreviation]*abbreviations_efficiency_odd[abbreviation] for abbreviation in abbreviations_efficiency_odd.keys() ], axis = 0 ) / np.sum( [ p0_arrays_odd[abbreviation] for abbreviation in abbreviations_efficiency_odd.keys() ], axis = 0 )
    peff_arrays_odd = np.mean( peff_arrays_odd, axis = (-2, -1) )


    exp_hpf_isotopes = np.loadtxt(data_dir_path / 'exp_data' / 'isotopes_hpf.dat')
    reduced_masses_experimental = np.array([red_mass_87Rb_84Sr_amu, red_mass_87Rb_86Sr_amu, red_mass_87Rb_87Sr_amu, red_mass_87Rb_88Sr_amu])
    peff_experiment = exp_hpf_isotopes[0,:]
    peff_std_experiment = exp_hpf_isotopes[1,:]
    #dpeff = 1e-3
    #p0_std = (p0(peff_experiment+dpeff/2, pmf_array=pmf_array)-p0(peff_experiment-dpeff/2, pmf_array=pmf_array))/dpeff * peff_std_experiment
    experiment = peff_experiment# if enhanced else p0(peff_experiment, pmf_array)
    std = peff_std_experiment# if enhanced else p0_std

    T_index = np.nonzero(temperatures == plot_temperature)[0][0]
    theory = np.moveaxis( [peff_arrays_even[:,T_index], peff_arrays_odd[:,T_index]], 0, -1)
    print(f'{theory=}')

    even_color = 'firebrick'
    odd_color = 'darkmagenta'
    theory_formattings = [ {'color': even_color, 'linewidth': 4},
                          {'color': odd_color, 'linewidth': 4}
                          ]
    theory_distinguished_formattings = [ {'color': 'k', 'linewidth': 4, 'linestyle':  (1.05,(0.1,2)), 'dash_capstyle': 'round' } for exp in experiment]
    experiment_formattings = [ {'color': 'firebrick', 'dash_capstyle': 'round', } for exp in experiment]

    fig_ax = fig.add_subplot()
    print(f'{reduced_masses_experimental=}, {experiment=}')
    fig_ax.scatter(reduced_masses_experimental[[0,1,3]], experiment[[0,1,3]], s = 16, c = 'firebrick', marker = 'd', edgecolors =  'firebrick', linewidths = None)
    fig_ax.errorbar(reduced_masses_experimental[[0,1,3]], experiment[[0,1,3]], std[[0,1,3]], ecolor = 'firebrick', capsize = 4, linestyle = 'None')
    fig_ax.scatter(reduced_masses_experimental[2], experiment[2], s = 16, c = 'darkmagenta', marker = 'd', edgecolors =  'darkmagenta', linewidths = None)
    fig_ax.errorbar(reduced_masses_experimental[2], experiment[2], std[2], ecolor = 'darkmagenta', capsize = 4, linestyle = 'None')
    fig_ax = ValuesVsModelParameters.plotValuestoAxis(fig_ax, reduced_masses, theory, experiment=None, std=None, theory_distinguished=None, theory_formattings = theory_formattings, theory_distinguished_formattings=theory_distinguished_formattings)
    fig_ax.set_ylim(0, 1.05*fig_ax.get_ylim()[1])
    PhaseTicks.linearStr(fig_ax.yaxis, 0.1, 0.05, '${x:.1f}$')
    PhaseTicks.linearStr(fig_ax.xaxis, 0.5, 0.1, '${x:.1f}$')

    for i, curve_name in enumerate(curves_names):
        # [print(f'{i}, {line.get_xydata()=}') for i, line in enumerate(fig_ax.get_lines())]
        if len(fig_ax.get_lines()[i+2*3].get_xydata()) > 0: fig_ax.get_lines()[i+2*3].set_label(curve_name) 
    xvals = (fig_ax.get_xlim()[0] + 0.1*(fig_ax.get_xlim()[1]-fig_ax.get_xlim()[0]), fig_ax.get_xlim()[0] + 0.6*(fig_ax.get_xlim()[1]-fig_ax.get_xlim()[0]))
    labelLines(fig_ax.get_lines(), xvals = xvals, align = False, outline_width=2, color = 'white', fontsize = matplotlib.rcParams["xtick.labelsize"], )
    labelLines(fig_ax.get_lines(), xvals = xvals, align = False, outline_color = None, yoffsets= -0*6.7e-3*(fig_ax.get_ylim()[1]-fig_ax.get_ylim()[0]), fontsize = matplotlib.rcParams["xtick.labelsize"], )
    # props = dict(boxstyle='round', facecolor='none', edgecolor='midnightblue')
    # fig_ax.text(0.03, 0.10, f'$\\Delta\\Phi_\\mathrm{{fit}} = {(magnetic_phases[0][1]-magnetic_phases[0][0])%1:.2f}\\pi$', va = 'center', ha = 'left', transform = fig_ax.transAxes, bbox = props)

    ylabel = f'$p_\mathrm{{eff}}$'# if enhanced else f'$p_0$'
    fig_ax.set_ylabel(ylabel)

    fig_ax.set_xlabel(f'reduced mass (a.m.u.)')

    return fig, fig_ax


def main():
    parser_description = "This is a python script for running molscat, collecting and pickling S-matrices, and calculating effective probabilities."
    parser = argparse.ArgumentParser(description=parser_description)
    parser.add_argument("-s", "--singlet_phase", type = float, default = None, help = "The singlet semiclassical phase modulo pi in multiples of pi.")
    parser.add_argument("-t", "--triplet_phase", type = float, default = None, help = "The triplet semiclassical phase modulo pi in multiples of pi.")
    parser.add_argument("-d", "--phase_difference", type = float, default = None, help = "The singlet-triplet semiclassical phase difference modulo pi in multiples of pi.")
    parser.add_argument("--mass_min", type = float, default = 42.5, help = "Minimum reduced mass for the grid (in a.m.u.).")
    parser.add_argument("--mass_max", type = float, default = 43.8, help = "Maximum reduced mass for the grid (in a.m.u.).")
    parser.add_argument("--dmass", type = float, default = 0.05, help = "Mass step (in a.m.u.).")

    parser.add_argument("--nenergies_barplot", type = int, default = 50, help = "Number of energy values in a grid.")
    parser.add_argument("--E_min_barplot", type = float, default = 8e-7, help = "Lowest energy value in the grid.")
    parser.add_argument("--E_max_barplot", type = float, default = 8e-2, help = "Highest energy value in the grid.")

    parser.add_argument("--nenergies_even", type = int, default = 50, help = "Number of energy values in a grid.")
    parser.add_argument("--E_min_even", type = float, default = 8e-7, help = "Lowest energy value in the grid.")
    parser.add_argument("--E_max_even", type = float, default = 8e-2, help = "Highest energy value in the grid.")

    parser.add_argument("--nenergies_odd", type = int, default = 50, help = "Number of energy values in a grid.")
    parser.add_argument("--E_min_odd", type = float, default = 4e-7, help = "Lowest energy value in the grid.")
    parser.add_argument("--E_max_odd", type = float, default = 4e-3, help = "Highest energy value in the grid.")

    parser.add_argument("--n_grid", type = int, default = 3, help = "n parameter for the nth-root energy grid.")
    parser.add_argument("-T", "--temperatures", nargs='*', type = float, default = None, help = "Temperature in the Maxwell-Boltzmann distributions (in kelvins).")

    parser.add_argument("--barplot_input_dir_name", type = str, default = 'RbSr+_tcpld_80mK_0.01_step', help = "Name of the directory with the molscat inputs")
    parser.add_argument("--barplot_SE_input_dir_name", type = str, default = 'RbSr+_tcpld_80mK_0.01_step_SE', help = "Name of the directory with the molscat inputs")
    parser.add_argument("--vs_mass_even_input_dir_name", type = str, default = 'RbSr+_tcpld_80mK_vs_mass', help = "Name of the directory with the molscat inputs")
    parser.add_argument("--vs_mass_odd_input_dir_name", type = str, default = 'RbSr+_tcpld_vs_mass_odd', help = "Name of the directory with the molscat inputs")

    parser.add_argument("--journal", type = str, default = 'NatCommun', help = "Name of the journal to prepare the plots for.")
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

    so_scaling_value = 0.375

    # reduced_masses = np.linspace(args.mass_min, args.mass_max, args.nmasses)
    reduced_masses = np.arange(args.mass_min, args.mass_max+0.5*args.dmass, args.dmass)

    if args.temperatures is None:
        temperatures = list(np.logspace(-4, -2, 20))
        temperatures.append(5e-4)
        temperatures = np.array(sorted(temperatures))
    else:
        temperatures = np.array(args.temperatures)

    [plotFig2(singlet_phase, triplet_phase, so_scaling_value, reduced_masses, energy_tuple_barplot= energy_tuple_barplot, energy_tuple_vs_mass_even = energy_tuple_vs_mass_even, energy_tuple_vs_mass_odd = energy_tuple_vs_mass_odd, temperatures = temperatures, plot_temperature = temperature, barplot_input_dir_name = args.barplot_input_dir_name, barplot_SE_input_dir_name = args.barplot_SE_input_dir_name, vs_mass_even_input_dir_name = args.vs_mass_even_input_dir_name, vs_mass_odd_input_dir_name = args.vs_mass_odd_input_dir_name, journal_name = args.journal) for temperature in temperatures]

if __name__ == '__main__':
    main()