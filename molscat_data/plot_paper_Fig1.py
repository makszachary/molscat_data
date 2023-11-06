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

def plotFig1(singlet_phases: float | np.ndarray[float], phase_differences: np.ndarray[float], singlet_phase_distinguished: float, so_phases: tuple[float, float], so_scaling_values: np.ndarray[float], energy_tuple: tuple[float, ...], temperatures: tuple[float, ...] = (5e-4,), plot_temperature: float = 5e-4, DPhi_input_dir_name: str = 'RbSr+_tcpld_80mK_0.01_step', SO_input_dir_name = 'RbSr+_tcpld_so_scaling', journal_name = 'NatCommun'):
    plt.style.use(Path(__file__).parent / 'mpl_style_sheets' / f'{journal_name}.mplstyle')
    nenergies = len(energy_tuple)
    E_min = min(energy_tuple)
    E_max = max(energy_tuple)

    png_path = plots_dir_path / 'paper' / f'{journal_name}' / 'Fig1' / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'Fig1_{plot_temperature:.2e}K.png'
    pdf_path = png_path.with_suffix('.pdf')
    svg_path = png_path.with_suffix('.svg')
    png_path.parent.mkdir(parents = True, exist_ok = True)

    cm = 1/2.54
    ws, hs = 0.05, 0.05
    nrows = 2
    row_height = 4.5
    vpad = 1
    total_height = nrows*row_height + (nrows-1)*vpad
    figsize = (18*cm, total_height*cm)
    dpi = 1000
    fig = plt.figure(figsize = figsize, dpi = dpi)
    gs_Figure = gridspec.GridSpec(nrows, 2, fig, hspace = hs, wspace = ws, height_ratios = [1 for row in range(nrows)], width_ratios = [135,45])    

    figs = [fig.add_subfigure(gs_Figure[:,0]), fig.add_subfigure(gs_Figure[0,1]), fig.add_subfigure(gs_Figure[1,1])]
    figs_axes = [[] for fig in figs]

    figs_axes[0].append(figs[0].add_subplot())

    
    figs_axes[1].append(figs[1].add_subplot())
    _temp_so_scal = 0.375
    figs_axes[1][0], _ax_chisq = plotPeffVsDPhiToAxis(figs_axes[1][0], singlet_phases = singlet_phases, phase_differences = phase_differences, so_scaling = _temp_so_scal, energy_tuple = energy_tuple, singlet_phase_distinguished = singlet_phase_distinguished, temperatures = temperatures, plot_temperature = plot_temperature, input_dir_name = DPhi_input_dir_name, hybrid = False)
    figs_axes[1].append(_ax_chisq)


    figs_axes[2].append(figs[2].add_subplot())
    # TEMPORARY TEMPORARY TEMPORARY
    _energy_tuple = tuple( round(n_root_scale(i, 4e-7, 4e-3, 50-1, n = 3), sigfigs = 11) for i in range(nenergies) )
    _temperatures = list(np.logspace(-4, -3, 10))
    _temperatures.append(plot_temperature)
    figs_axes[2][0] = plotPeffVsSOScalingToAxis(figs_axes[2][0], so_scaling_values = so_scaling_values, singlet_phase = so_phases[0], triplet_phase = so_phases[1], energy_tuple = _energy_tuple, temperatures = _temperatures, plot_temperature = plot_temperature, input_dir_name = SO_input_dir_name)

    figs_axes[0][0].text(0., 1.04, f'a', fontsize = 7, family = 'sans-serif', va = 'top', ha = 'left', transform = fig.transFigure, fontweight = 'bold')
    figs_axes[1][0].text(0.7, 1.04, f'b', fontsize = 7, family = 'sans-serif', va = 'top', ha = 'left', transform = fig.transFigure, fontweight = 'bold')
    figs_axes[2][0].text(0.7, 0.54, f'c', fontsize = 7, family = 'sans-serif', va = 'top', ha = 'left', transform = fig.transFigure, fontweight = 'bold')

    figs[0].subplots_adjust(left = 0.05, bottom = 0.2, top = 0.98)
    figs[1].subplots_adjust(left = 0.05, bottom = 0.2, top = 0.96)
    figs[2].subplots_adjust(left = 0.05, bottom = 0.2, top = 0.96)

    fig.savefig(png_path, bbox_inches='tight', pad_inches = 0)
    fig.savefig(svg_path, bbox_inches='tight', pad_inches = 0, transparent = True)
    fig.savefig(pdf_path, bbox_inches='tight', pad_inches = 0, transparent = True)

    plt.close()
    

def plotPeffVsDPhiToAxis(ax, singlet_phases: float | np.ndarray[float], phase_differences: float | np.ndarray[float], so_scaling: float, energy_tuple: tuple[float, ...], singlet_phase_distinguished: float = None, temperatures: tuple[float, ...] = (5e-4,), plot_temperature: float = 5e-4, input_dir_name: str = 'RbSr+_tcpld_80mK', hybrid = False):
    nenergies = len(energy_tuple)
    E_min = min(energy_tuple)
    E_max = max(energy_tuple)
    singlet_phases, phase_differences = np.array(singlet_phases), np.array(phase_differences)
    probabilities_dir_name = 'probabilities_hybrid' if hybrid else 'probabilities'

    array_paths_hot = [ [arrays_dir_path / input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{(singlet_phase+phase_difference)%1:.4f}' / f'{so_scaling:.4f}' / probabilities_dir_name / 'hpf.txt' if ( singlet_phase+phase_difference ) % 1 !=0 else None for phase_difference in phase_differences ] for singlet_phase in singlet_phases]
    array_paths_cold_higher = [  [arrays_dir_path / input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{(singlet_phase+phase_difference)%1:.4f}' / f'{so_scaling:.4f}' / probabilities_dir_name / 'cold_higher.txt' if ( singlet_phase+phase_difference ) % 1 !=0 else None for phase_difference in phase_differences] for singlet_phase in singlet_phases]
    # [ print( np.loadtxt(array_path).shape ) if array_path is not None else np.full((len(temperatures), 5), np.nan) for sublist in array_paths_hot for array_path in sublist ]

    arrays_hot = np.array([ [np.loadtxt(array_path) if (array_path is not None and array_path.is_file()) else np.full((len(temperatures), 5), np.nan) for array_path in sublist] for sublist in array_paths_hot ])
    arrays_hot = arrays_hot.reshape(*arrays_hot.shape[0:2], len(temperatures), -1)

    arrays_cold_higher = np.array( [ [np.loadtxt(array_path) if (array_path is not None and array_path.is_file()) else np.full((len(temperatures), 5), np.nan) for array_path in sublist] for sublist in array_paths_cold_higher ] )
    arrays_cold_higher = arrays_cold_higher.reshape(*arrays_cold_higher.shape[0:2], len(temperatures), -1)

    singlet_phases = np.full((len(phase_differences), len(singlet_phases)), singlet_phases).transpose()

    if singlet_phase_distinguished is not None:
        array_paths_hot_distinguished = [arrays_dir_path / input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase_distinguished:.4f}_{(singlet_phase_distinguished+phase_difference)%1:.4f}' / f'{so_scaling:.4f}' / probabilities_dir_name / 'hpf.txt' if ( singlet_phase_distinguished+phase_difference ) % 1 !=0 else None for phase_difference in phase_differences]
        array_paths_cold_higher_distinguished = [arrays_dir_path / input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase_distinguished:.4f}_{(singlet_phase_distinguished+phase_difference)%1:.4f}' / f'{so_scaling:.4f}' / probabilities_dir_name / 'cold_higher.txt' if ( singlet_phase_distinguished+phase_difference ) % 1 !=0 else None for phase_difference in phase_differences]
        arrays_hot_distinguished = np.array( [np.loadtxt(array_path) if (array_path is not None and array_path.is_file()) else np.full((len(temperatures), 5), np.nan) for array_path in array_paths_hot_distinguished ] )
        arrays_hot_distinguished = arrays_hot_distinguished.reshape(arrays_hot_distinguished.shape[0], len(temperatures), -1)
        arrays_cold_higher_distinguished = np.array( [np.loadtxt(array_path) if (array_path is not None and array_path.is_file()) else np.full((len(temperatures), 5), np.nan) for array_path in array_paths_cold_higher_distinguished ] )
        arrays_cold_higher_distinguished = arrays_cold_higher_distinguished.reshape(arrays_cold_higher_distinguished.shape[0], len(temperatures), -1)
    

    exp_hot = np.loadtxt(data_dir_path / 'exp_data' / 'single_ion_hpf.dat')
    exp_cold_higher = np.loadtxt(data_dir_path / 'exp_data' / 'single_ion_cold_higher.dat')
    experiment = np.array( [ exp_hot[0,0], exp_cold_higher[0,0] ] )
    std = np.array( [ exp_hot[1,0], exp_cold_higher[1,0] ] )

    xx = np.full((len(singlet_phases), len(phase_differences)), phase_differences).transpose()
    T_index = np.nonzero(temperatures == plot_temperature)[0][0]
    theory_distinguished = np.moveaxis(np.array( [[ arrays_hot_distinguished[:,T_index,0],], [arrays_cold_higher_distinguished[:,T_index,0], ]] ), 0, -1)
    theory = np.moveaxis(np.array( [ arrays_hot[:,:,T_index,0], arrays_cold_higher[:,:,T_index,0] ] ), 0, -1) if (singlet_phase_distinguished is not None) else theory_distinguished
    
    chi_sq_distinguished = chi_squared(theory_distinguished, experiment, std)
    minindex_distinguished = np.nanargmin(chi_sq_distinguished)
    xx_min_distinguished = xx[:,1][minindex_distinguished]
    chi_sq_min_distinguished = np.nanmin(chi_sq_distinguished)

    theory_formattings = [ {'color': 'darksalmon', 'linewidth': 0.02},
                          {'color': 'lightsteelblue', 'linewidth': 0.02} ]
    theory_distinguished_formattings = [ {'color': 'firebrick', 'linewidth': 1.5},
                        {'color': 'midnightblue', 'linewidth': 1.5} ]
    experiment_formattings = [ {'color': 'firebrick', 'linewidth': 1.5, 'linestyle': '--'},
                        {'color': 'midnightblue', 'linewidth': 1.5, 'linestyle': '--'} ]

    ax, ax_chisq = ValuesVsModelParameters.plotValuesAndChiSquaredToAxis(ax, xx, theory, experiment, std, theory_distinguished, theory_formattings = theory_formattings, theory_distinguished_formattings = theory_distinguished_formattings, experiment_formattings = experiment_formattings, )
    data = np.array([line.get_xydata() for line in ax_chisq.lines])
    # minindices = np.nanargmin(data[:,:,1])
    # xx_min = xx[minindices]
    chi_sq_min = np.nanmin(data[:,:,1], axis=1)
    ax.set_ylim(0,1)

    PhaseTicks.setInMultiplesOfPhi(ax.xaxis)

    PhaseTicks.linearStr(ax.yaxis, 0.2, 0.1, '${x:.1f}$')

    ax.set_xlabel(f'$\\Delta\\Phi$')
    ax.set_ylabel(f'$p_\\mathrm{{eff}}$')
    ax_chisq.set_ylabel(f'$\\chi^2$', rotation = 0, labelpad = 8)
    
    ax.xaxis.get_major_ticks()[1].label1.set_visible(False)
    ax_chisq.legend(loc = 'upper left', handletextpad=0.2, frameon=False)

    return ax, ax_chisq

def plotPeffVsSOScalingToAxis(ax, so_scaling_values, singlet_phase, triplet_phase, energy_tuple: tuple[float, ...], temperatures: tuple[float, ...] = (5e-4,), plot_temperature: float = 5e-4, input_dir_name: str = 'RbSr+_tcpld_so_scaling',):
    print('YS0')
    nenergies = len(energy_tuple)
    E_min = min(energy_tuple)
    E_max = max(energy_tuple)

    array_paths_hot = [ arrays_dir_path / input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{triplet_phase:.4f}' / f'{so_scaling:.4f}' / 'in_4_4_1_1' / 'probabilities' / 'hpf.txt' for so_scaling in so_scaling_values ]
    arrays_hot = np.array([ np.loadtxt(array_path) if (array_path is not None and array_path.is_file()) else np.full((len(temperatures), 5), np.nan) for array_path in array_paths_hot ])
    arrays_hot = arrays_hot.reshape(*arrays_hot.shape[0:1], len(temperatures), -1)
  

    exp_hot = np.loadtxt(data_dir_path / 'exp_data' / 'single_ion_hpf.dat')
    exp_cold_higher = np.loadtxt(data_dir_path / 'exp_data' / 'single_ion_cold_higher.dat')
    experiment = np.array( [ exp_hot[0,0], ] )
    std = np.array( [ exp_hot[1,0], ] )

    # xx = np.full((len(so_scaling_values), 1), so_scaling_values).transpose()
    xx = np.array(so_scaling_values)
    T_index = np.nonzero(temperatures == plot_temperature)[0][0]
    theory = np.moveaxis(np.array( [ arrays_hot[:,T_index,0], ] ), 0, -1)
    theory_distinguished = theory
    
    print('YS')
    theory_formattings = [ {'color': 'darksalmon', 'linewidth': 0.02}, ]
    theory_distinguished_formattings = [ {'color': 'firebrick', 'linewidth': 1.5}, ]
    experiment_formattings = [ {'color': 'firebrick', 'linewidth': 1.5, 'linestyle': '--'},
                        {'color': 'midnightblue', 'linewidth': 1.5, 'linestyle': '--'} ]

    # array_paths = ( arrays_dir_path / 'data_produced' / 'arrays' / f'{input_dir_name}' / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{triplet_phase:.4f}' / f'{so_scaling:.4f}' / 'in_4_4_1_1' / 'probabilities' / 'hpf.txt' for so_scaling in so_scaling_values )
    # output_state_resolved_arrays = list( np.loadtxt(array_path) for array_path in array_paths )
    # pmf_path = Path(__file__).parents[1] / 'data' / 'pmf' / 'N_pdf_logic_params_EMM_500uK.txt'
    # pmf_array = np.loadtxt(pmf_path)

    # p_eff_exp = 0.0600
    # p_eff_exp_std = 0.0227
    # ss_dominated_rates = np.fromiter( (array[4] for array in output_state_resolved_arrays), dtype = float )

    # fig, ax = ProbabilityVersusSpinOrbit.plotEffectiveProbability(so_scaling_values, ss_dominated_rates, p_eff_exp=p_eff_exp, p_eff_exp_std=p_eff_exp_std, pmf_array = pmf_array)
    
    ax = ValuesVsModelParameters.plotValuestoAxis(ax, xx, theory, experiment, std, theory_formattings = theory_distinguished_formattings)
    ax.scatter(so_scaling_values, theory.flatten(), s = 6**2, color = 'k', marker = 'o')

    return ax


def main():
    parser_description = "This is a python script for running molscat, collecting and pickling S-matrices, and calculating effective probabilities."
    parser = argparse.ArgumentParser(description=parser_description)
    parser.add_argument("-d", "--phase_step", type = float, default = None, help = "The phase step multiples of pi.")
    parser.add_argument("-s", "--singlet_phase", type = float, default = None, help = "The distinguished value of the singlet semiclassical phase modulo pi in multiples of pi.")
    parser.add_argument("-t", "--triplet_phase", type = float, default = None, help = "The distinguished value of the triplet semiclassical phase modulo pi in multiples of pi.")
    parser.add_argument("--so_scaling", nargs='*', type = float, default = [0.325,], help = "Values of the SO scaling.")
    parser.add_argument("--nenergies", type = int, default = 50, help = "Number of energy values in a grid.")
    parser.add_argument("--E_min", type = float, default = 4e-7, help = "Lowest energy value in the grid.")
    parser.add_argument("--E_max", type = float, default = 4e-3, help = "Highest energy value in the grid.")
    parser.add_argument("--n_grid", type = int, default = 3, help = "n parameter for the nth-root energy grid.")
    parser.add_argument("-T", "--temperatures", nargs='*', type = float, default = None, help = "Temperature in the Maxwell-Boltzmann distributions (in kelvins).")
    parser.add_argument("--DPhi_input_dir_name", type = str, default = 'RbSr+_fmf_DPhi_SE', help = "Name of the directory with the molscat inputs")
    parser.add_argument("--SO_input_dir_name", type = str, default = 'RbSr+_fmf_so_scaling', help = "Name of the directory with the molscat inputs")
    parser.add_argument("--journal", type = str, default = 'NatCommun', help = "Name of the journal to prepare the plots for.")
    args = parser.parse_args()

    nenergies, E_min, E_max, n = args.nenergies, args.E_min, args.E_max, args.n_grid
    energy_tuple = tuple( round(n_root_scale(i, E_min, E_max, nenergies-1, n = n), sigfigs = 11) for i in range(nenergies) )

    singlet_phase_distinguished = args.singlet_phase if args.singlet_phase is not None else default_singlet_phase_function(1.0)
    triplet_phase_distinguished = args.triplet_phase if args.triplet_phase is not None else default_triplet_phase_function(1.0)

    singlet_phases = np.array([default_singlet_phase_function(1.0),]) if args.phase_step is None else np.arange(args.phase_step, 1., args.phase_step).round(decimals=4)
    triplet_phases = np.array([default_triplet_phase_function(1.0),]) if args.phase_step is None else np.arange(args.phase_step, 1., args.phase_step).round(decimals=4)
    phase_differences = np.arange(0, 1.+args.phase_step, args.phase_step).round(decimals=4)
    so_scaling_values = list(set(args.so_scaling))

    if args.temperatures is None:
        temperatures = list(np.logspace(-4, -2, 20))
        temperatures.append(5e-4)
        temperatures = np.array(sorted(temperatures))
    else:
        temperatures = np.array(args.temperatures)


    [plotFig1(singlet_phases = singlet_phases, phase_differences = phase_differences, singlet_phase_distinguished = singlet_phase_distinguished, so_phases = (singlet_phase_distinguished, triplet_phase_distinguished), so_scaling_values = so_scaling_values, energy_tuple = energy_tuple, temperatures = temperatures, plot_temperature = temperature, DPhi_input_dir_name = args.DPhi_input_dir_name, SO_input_dir_name = args.SO_input_dir_name, journal_name = args.journal) for temperature in temperatures]

if __name__ == '__main__':
    main()