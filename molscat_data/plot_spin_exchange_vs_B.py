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

def plot_probability_vs_B(phases: tuple[tuple[float, float], ...], phases_distinguished: tuple[float, float], magnetic_fields: float | np.ndarray[float], magnetic_field_experimental: float, MF_in: int, MS_in: int, energy_tuple: tuple[float, ...], temperatures: tuple[float, ...] = (5e-4,), plot_temperature: float = 5e-4, input_dir_name: str = 'RbSr+_fmf_SE_vs_B_80mK', enhanced = False, journal_name = 'NatCommun'):
    plt.style.use(Path(__file__).parent / 'mpl_style_sheets' / f'{journal_name}.mplstyle')

    nenergies = len(energy_tuple)
    E_min = min(energy_tuple)
    E_max = max(energy_tuple)
    probabilities_dir_name = 'probabilities'
    prefix_for_array_path = '' if enhanced else 'p0_'
    F1, F2 = 2, 1
    MF1, MF2 = MF_in, MS_in

    abbreviation='cold'
    
    array_paths_cold_lower = [  [arrays_dir_path / input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{triplet_phase:.4f}' / f'{F1}_{MF1}_{F2}_{MF2}' / f'{magnetic_field:.2f}' / probabilities_dir_name / f'{prefix_for_array_path}{abbreviation}.txt' for magnetic_field in magnetic_fields] for singlet_phase, triplet_phase in phases]
    [ [print(array_path) for array_path in sublist if (array_path is not None and not array_path.is_file())] for sublist in array_paths_cold_lower ]
    arrays_cold_lower = np.array([ [np.loadtxt(array_path) if (array_path is not None and array_path.is_file()) else np.full((len(temperatures), 3), np.nan) for array_path in sublist] for sublist in array_paths_cold_lower ])
    arrays_cold_lower = arrays_cold_lower.reshape(*arrays_cold_lower.shape[:2], len(temperatures), -1)

    # if phases_distinguished is not None:
    #     array_paths_cold_lower_distinguished = [arrays_dir_path / input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{phases_distinguished[0]:.4f}_{phases_distinguished[1]:.4f}' / f'{F1}_{MF1}_{F2}_{MF2}' / f'{magnetic_field:.2f}' / probabilities_dir_name / f'{prefix_for_array_path}{abbreviation}.txt' for magnetic_field in magnetic_fields]
    #     arrays_cold_lower_distinguished = np.array([ np.loadtxt(array_path) if (array_path is not None and array_path.is_file()) else np.full((len(temperatures), 3), np.nan) for array_path in array_paths_cold_lower_distinguished ])
    #     arrays_cold_lower_distinguished = arrays_cold_lower_distinguished.reshape(arrays_cold_lower_distinguished.shape[0], len(temperatures), -1)

    # exp_hot = np.loadtxt(data_dir_path / 'exp_data' / 'single_ion_hpf.dat')
    # exp_cold_higher = np.loadtxt(data_dir_path / 'exp_data' / 'single_ion_cold_higher.dat')
    exp_cold_lower = np.loadtxt(data_dir_path / 'exp_data' / 'single_ion_cold_lower.dat')

    peff_experiment = np.array([exp_cold_lower[0,0],])
    peff_std_experiment = np.array([exp_cold_lower[1,0],])
    dpeff = 1e-3
    p0_std = (p0(peff_experiment+dpeff/2, pmf_array=pmf_array)-p0(peff_experiment-dpeff/2, pmf_array=pmf_array))/dpeff * peff_std_experiment
    experiment = peff_experiment if enhanced else p0(peff_experiment, pmf_array)
    std = peff_std_experiment if enhanced else p0_std

    T_index = np.nonzero(temperatures == plot_temperature)[0][0]
    theory = np.moveaxis( arrays_cold_lower[:,:,T_index,0], 0, -1)
    # print(theory.shape)
    # theory_distinguished = np.moveaxis(np.array( [ arrays_cold_lower_distinguished[:,T_index, 0], ]), 0, -1)
    theory_distinguished = None

    prefix_for_image_path = 'peff_' if enhanced else 'p0_'
    png_path = plots_dir_path / 'paper' / f'{journal_name}' / f'{prefix_for_image_path}f=1_SE_vs_B' / f'{input_dir_name}' / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'SE_{prefix_for_image_path}vs_B_{plot_temperature:.2e}K.png'
    pdf_path = png_path.with_suffix('.pdf')
    svg_path = png_path.with_suffix('.svg')
    png_path.parent.mkdir(parents = True, exist_ok = True)

    # color_map = matplotlib.colormaps['twilight']
    color_map = cmcrameri.cm.devon
    theory_colors = list(reversed([color_map(singlet_phase) for singlet_phase, triplet_phase in phases]))
    theory_formattings = [ {'color': color, 'linewidth': 2} for color in theory_colors ]
    theory_distinguished_formattings = [ {'color': 'k', 'linewidth': 4, 'linestyle':  (0.8,(0.1,2)), 'dash_capstyle': 'round' } for exp in experiment]
    # theory_distinguished_formattings = [ {'color': 'k', 'linewidth': 4, 'linestyle':  '--' } for exp in experiment]

    cm = 1/2.54
    figsize = (18.5*cm, 6*cm)
    dpi = 1000
    # fig, ax0 = ValuesVsModelParameters.plotValues(magnetic_fields, theory, experiment, std, theory_distinguished, theory_colors, theory_distinguished_colors, figsize=figsize, dpi=dpi)
    fig, ax0 = ValuesVsModelParameters.plotValues(magnetic_fields, theory, experiment=None, std=None, theory_distinguished=None, theory_formattings = theory_formattings, theory_distinguished_formattings=theory_distinguished_formattings, figsize=figsize, dpi=dpi)
    ax0.scatter([magnetic_field_experimental,], experiment, s = 16, c = theory_distinguished_formattings[0]['color'], marker = 'd')
    # print(std)
    ax0.errorbar([magnetic_field_experimental, ], experiment, std, ecolor = theory_distinguished_formattings[0]['color'], capsize = 6)
    ax0.set_ylim(0, 1.05*ax0.get_ylim()[1])
    PhaseTicks.linearStr(ax0.yaxis, 0.1, 0.05, '${x:.1f}$')
    PhaseTicks.linearStr(ax0.xaxis, 50, 10, '${x:n}$')
    for i, (singlet_phase, triplet_phase) in enumerate(phases):
        ax0.get_lines()[i].set_label(f'$\\Phi_\\mathrm{{s}} = {singlet_phase:.2f}\\pi$')
    # props = dict(boxstyle='round,pad=-0.05, rounding_size=0.2', facecolor='white', edgecolor='none')
    labelLines(ax0.get_lines(), align = False, outline_width=2, fontsize = 'small', color = 'white')
    labelLines(ax0.get_lines(), align = False, outline_width=2, outline_color = None, yoffsets= -6.7e-3*(ax0.get_ylim()[1]-ax0.get_ylim()[0]), fontsize = 'small')
    props = dict(boxstyle='round', facecolor='none', edgecolor='midnightblue')
    ax0.text(0.03, 0.10, f'$\\Delta\\Phi_\\mathrm{{fit}} = {(phases[0][1]-phases[0][0])%1:.2f}\\pi$', va = 'center', ha = 'left', transform = ax0.transAxes, bbox = props)

    # color_map = matplotlib.colormaps['plasma'] or 'inferno'
    color_map = cmocean.cm.thermal
    lognorm = matplotlib.colors.LogNorm(vmin=min(temperatures), vmax=max(temperatures), clip = False)
    theory_colors = [color_map(lognorm(temperature)) for temperature in temperatures[::2]]
    theory_formattings = [ {'color': color, 'linewidth': 1.5} for color in theory_colors ]
    # theory_distinguished_colors = ['firebrick', ]

    T_index = np.nonzero(temperatures == plot_temperature)[0][0]    

    gs = gridspec.GridSpec(3,185)
    ax0.set_position(gs[:,:120].get_position(fig))
    ax0.set_subplotspec(gs[:,:120])

    theory = arrays_cold_lower[0,:,::2,0]
    theory_distinguished = np.moveaxis( np.array( [arrays_cold_lower[0,:,T_index,0],]), 0, -1)
    ax1 = fig.add_subplot(gs[0,131:181], sharex = ax0)
    ax1 = ValuesVsModelParameters.plotValuestoAxis(ax1, magnetic_fields, theory, None, None, theory_distinguished, theory_formattings = theory_formattings, theory_distinguished_formattings=theory_distinguished_formattings)
    ax1.set_ylim(0, ax1.get_ylim()[1])


    theory = arrays_cold_lower[1,:,::2,0]
    theory_distinguished = np.moveaxis( np.array( [arrays_cold_lower[1,:,T_index,0],]), 0, -1)

    ax2 = fig.add_subplot(gs[1,131:181], sharex = ax0)
    ax2 = ValuesVsModelParameters.plotValuestoAxis(ax2, magnetic_fields, theory, None, None, theory_distinguished, theory_formattings = theory_formattings, theory_distinguished_formattings=theory_distinguished_formattings)
    ax2.set_ylim(0, ax2.get_ylim()[1])

    theory = arrays_cold_lower[2,:,::2,0]
    theory_distinguished = np.moveaxis( np.array( [arrays_cold_lower[2,:,T_index,0],]), 0, -1)

    ax3 = fig.add_subplot(gs[2,131:181], sharex = ax0)
    ax3 = ValuesVsModelParameters.plotValuestoAxis(ax3, magnetic_fields, theory, None, None, theory_distinguished, theory_formattings = theory_formattings, theory_distinguished_formattings=theory_distinguished_formattings)
    ax3.set_ylim(0, ax3.get_ylim()[1])

    PhaseTicks.linearStr(ax1.yaxis, 0.2, 0.1, '${x:.1f}$')
    ax1.yaxis.set_major_formatter(ticker.StrMethodFormatter('${x:.1f}$'))
    PhaseTicks.linearStr(ax2.yaxis, 0.2, 0.1, '${x:.1f}$')
    ax2.yaxis.set_major_formatter(ticker.StrMethodFormatter('${x:.1f}$'))
    PhaseTicks.linearStr(ax3.yaxis, 0.1, 0.05, '${x:.1f}$')
    ax3.yaxis.set_major_formatter(ticker.StrMethodFormatter('${x:.1f}$'))
    

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    ax2.yaxis.get_major_ticks()[-1].label1.set_visible(False)
    ax3.yaxis.get_major_ticks()[-1].label1.set_visible(False)
    
    ax0.set_xlabel(f'$B\\,(\\mathrm{{G}})$')
    ax3.set_xlabel(f'$B\\,(\\mathrm{{G}})$')
    ylabel = f'$p_\mathrm{{eff}}$' if enhanced else f'$p_0$'
    ax0.set_ylabel(ylabel)#, rotation = 0, lapelpad = 12)

    # create the temperature bar
    ax1_bar = fig.add_subplot(gs[:,182:])

    bar_format = theory_distinguished_formattings[0].copy()
    # bar_format['linewidth'] = 1
    # bar_format['linestyle'] =  (1,(0.1,1))

    bar = matplotlib.colorbar.ColorbarBase(ax1_bar, cmap = color_map, norm = lognorm, ticks = [1e-4, plot_temperature, 1e-3, 1e-2], )
    bar.set_ticklabels(['$0.1$', f'$T_\\mathrm{{exp}}$', '$1$', '$10$'])
    bar.ax.axhline(plot_temperature, **bar_format)
    ax1_bar.tick_params(axis = 'both')
    ax1_bar.get_yaxis().labelpad = 4
    ax1_bar.set_ylabel('$T\\,(\\mathrm{mK})$', rotation = 0, va = 'baseline', ha = 'left')
    ax1_bar.yaxis.set_label_coords(0.0, 1.05)


    ax0.text(-0.09, 1.11, f'c', family = 'sans-serif', fontsize = 7, va = 'top', ha = 'left', transform = ax0.transAxes, fontweight = 'bold')
    ax1.text(-0.14, 1.33, f'd', family = 'sans-serif', fontsize = 7, va = 'top', ha = 'left', transform = ax1.transAxes, fontweight = 'bold')
    fig.subplots_adjust(left = 0.07, top = 0.9, right = 0.95, bottom = 0.20, hspace = .0)
    # fig.tight_layout()
    fig.savefig(png_path)#, bbox_inches='tight')#, pad_inches = 0.)
    fig.savefig(svg_path, transparent = True)#, bbox_inches='tight')#, pad_inches = 0.)
    fig.savefig(pdf_path, transparent = True)
    plt.close()

def main():
    parser_description = "This is a python script for running molscat, collecting and pickling S-matrices, and calculating effective probabilities."
    parser = argparse.ArgumentParser(description=parser_description)
    parser.add_argument("-s", "--singlet_phases", nargs='*', type = float, default = [0.04,], help = "The singlet semiclassical phase modulo pi in multiples of pi.")
    parser.add_argument("-t", "--triplet_phases", nargs='*', type = float, default = [0.23,], help = "The triplet semiclassical phase modulo pi in multiples of pi.")
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
    parser.add_argument("--input_dir_name", type = str, default = 'RbSr+_fmf_SE_vs_B_80mK', help = "Name of the directory with the molscat inputs")
    parser.add_argument("--enhanced", action = 'store_true', help = "If enabled, the effective probabilities will be plotted instead of the short-range p0.")
    parser.add_argument("--journal", type = str, default = 'NatCommun', help = "Name of the journal to prepare the plots for")
    args = parser.parse_args()

    F1, MF1, F2, MF2 = 2, args.MF_in, 1, args.MS_in

    nenergies, E_min, E_max, n = args.nenergies, args.E_min, args.E_max, args.n_grid
    energy_tuple = tuple( round(n_root_scale(i, E_min, E_max, nenergies-1, n = n), sigfigs = 11) for i in range(nenergies) )
    phases = tuple(zip(list(args.singlet_phases), list(args.triplet_phases)))
    magnetic_fields = np.arange(args.B_min, args.B_max+0.1*args.dB, args.dB)

    if args.temperatures is None:
        temperatures = list(np.logspace(-4, -2, 20))
        temperatures.append(5e-4)
        temperatures = np.array(sorted(temperatures))
    else:
        temperatures = np.array(args.temperatures)

    [plot_probability_vs_B(phases = phases, phases_distinguished= (0.04, 0.23), magnetic_fields = magnetic_fields, magnetic_field_experimental = 3., MF_in = MF1, MS_in = MF2, energy_tuple = energy_tuple, temperatures = temperatures, plot_temperature = temperature, input_dir_name = args.input_dir_name, enhanced = args.enhanced, journal_name = args.journal) for temperature in temperatures]

if __name__ == '__main__':
    main()