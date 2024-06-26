import argparse
import os
import sys
from pathlib import Path

import numpy as np
from sigfig import round

from matplotlib import pyplot as plt

from scipy.optimize import curve_fit, brute, differential_evolution

from _molscat_data.thermal_averaging import n_root_scale
from _molscat_data.effective_probability import effective_probability, p0
from _molscat_data.scaling_old import parameter_from_semiclassical_phase, semiclassical_phase_function, default_singlet_parameter_from_phase, default_triplet_parameter_from_phase, default_singlet_phase_function, default_triplet_phase_function

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


def get_p0_vs_DPhi(singlet_phases: float | np.ndarray[float], phase_differences: float | np.ndarray[float], so_scaling: float, energy_tuple: tuple[float, ...], singlet_phase_distinguished: float = None, temperatures: tuple[float, ...] = (5e-4,), plot_temperature: float = 5e-4, input_dir_name: str = 'RbSr+_tcpld_80mK',):
    plot_p0 = True
    nenergies = len(energy_tuple)
    E_min = min(energy_tuple)
    E_max = max(energy_tuple)
    singlet_phases, phase_differences = np.array(singlet_phases), np.array(phase_differences)
    probabilities_dir_name = 'probabilities'

    array_paths_hot = [ [arrays_dir_path / input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{(singlet_phase+phase_difference)%1:.4f}' / f'{so_scaling:.4f}' / 'in_4_-4_1_1' / probabilities_dir_name / ('p0_hpf.txt' if plot_p0 else 'hpf.txt') if ( singlet_phase+phase_difference ) % 1 !=0 else None for phase_difference in phase_differences ] for singlet_phase in singlet_phases]
    array_paths_cold_higher = [  [arrays_dir_path / input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{(singlet_phase+phase_difference)%1:.4f}' / f'{so_scaling:.4f}' / 'in_4_-4_1_1' / probabilities_dir_name / ('p0_cold_higher.txt' if plot_p0 else 'cold_higher.txt') if ( singlet_phase+phase_difference ) % 1 !=0 else None for phase_difference in phase_differences] for singlet_phase in singlet_phases]
    if not np.loadtxt(array_paths_hot[0][0]).shape[-1] == len(temperatures):
        raise ValueError(f"{len(temperatures)=} should be equal to {np.loadtxt(array_paths_hot[0][0]).shape[-1]=}")
    arrays_hot = np.array([ [np.loadtxt(array_path) if (array_path is not None and array_path.is_file()) else np.full(np.loadtxt(array_paths_hot[0][0]).shape, np.nan) for array_path in sublist] for sublist in array_paths_hot ])
    arrays_hot = arrays_hot.reshape(*arrays_hot.shape[0:2], len(temperatures), -1).squeeze()

    if not np.loadtxt(array_paths_cold_higher[0][0]).shape[-1] == len(temperatures):
        raise ValueError(f"{len(temperatures)=} should be equal to {np.loadtxt(array_paths_cold_higher[0][0]).shape[-1]=}")
    arrays_cold_higher = np.array( [ [np.loadtxt(array_path) if (array_path is not None and array_path.is_file()) else np.full(np.loadtxt(array_paths_hot[0][0]).shape, np.nan) for array_path in sublist] for sublist in array_paths_cold_higher ] )
    arrays_cold_higher = arrays_cold_higher.reshape(*arrays_cold_higher.shape[0:2], len(temperatures), -1).squeeze()

    singlet_phases = np.full((len(phase_differences), len(singlet_phases)), singlet_phases).transpose()

    if singlet_phase_distinguished is not None:
        array_paths_hot_distinguished = [arrays_dir_path / input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase_distinguished:.4f}_{(singlet_phase_distinguished+phase_difference)%1:.4f}' / f'{so_scaling:.4f}' / 'in_4_-4_1_1' / probabilities_dir_name / ('p0_hpf.txt' if plot_p0 else 'hpf.txt') if ( singlet_phase_distinguished+phase_difference ) % 1 !=0 else None for phase_difference in phase_differences]
        array_paths_cold_higher_distinguished = [arrays_dir_path / input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase_distinguished:.4f}_{(singlet_phase_distinguished+phase_difference)%1:.4f}' / f'{so_scaling:.4f}' / 'in_4_-4_1_1' / probabilities_dir_name / ('p0_cold_higher.txt' if plot_p0 else 'cold_higher.txt') if ( singlet_phase_distinguished+phase_difference ) % 1 !=0 else None for phase_difference in phase_differences]
        arrays_hot_distinguished = np.array( [np.loadtxt(array_path) if (array_path is not None and array_path.is_file()) else np.full(np.loadtxt(array_paths_hot[0][0]).shape, np.nan) for array_path in array_paths_hot_distinguished ] )
        arrays_hot_distinguished = arrays_hot_distinguished.reshape(arrays_hot_distinguished.shape[0], len(temperatures), -1).squeeze()
        arrays_cold_higher_distinguished = np.array( [np.loadtxt(array_path) if (array_path is not None and array_path.is_file()) else np.full(np.loadtxt(array_paths_hot[0][0]).shape, np.nan) for array_path in array_paths_cold_higher_distinguished ] )
        arrays_cold_higher_distinguished = arrays_cold_higher_distinguished.reshape(arrays_cold_higher_distinguished.shape[0], len(temperatures), -1).squeeze()
    

    exp_hot = np.loadtxt(data_dir_path / 'exp_data' / ('p0_single_ion_hpf.dat' if plot_p0 else 'single_ion_hpf.dat') )
    exp_cold_higher = np.loadtxt(data_dir_path / 'exp_data' / ('p0_single_ion_cold_higher.dat' if plot_p0 else 'single_ion_cold_higher.dat') )
    experiment = np.array( [ exp_hot[0,0], exp_cold_higher[0,0] ] )
    std = np.array( [ exp_hot[1,0], exp_cold_higher[1,0] ] )

    
    xx = np.full((len(singlet_phases), len(phase_differences)), phase_differences).transpose()
    T_index = np.nonzero(temperatures == plot_temperature)[0][0]
    theory_distinguished = np.moveaxis(np.array( [[ arrays_hot_distinguished[:,T_index],], [arrays_cold_higher_distinguished[:,T_index], ]] ), 0, -1)
    theory = np.moveaxis(np.array( [ arrays_hot[:,:,T_index], arrays_cold_higher[:,:,T_index] ] ), 0, -1) if (singlet_phase_distinguished is not None) else theory_distinguished
    
    return phase_differences, theory_distinguished, experiment, std



def spin_exchange(DeltaPhi, Phi0, amplitude):
    spin_exchange = amplitude * np.sin(DeltaPhi + Phi0)**2
    return spin_exchange


def residuals(f, xdata, ydata, yerr = None, *args, **kwargs):
    xdata, ydata, yerr = np.asarray(xdata), np.asarray(ydata), np.asarray(yerr)
    if ydata.shape != xdata.shape: raise IndexError("Length of xdata and ydata should be the same.")
    if yerr is None: yerr = np.ones(ydata.shape)
    
    return np.array([(f(x, *args, **kwargs) - ydata[index]) / yerr[index] for index, x in np.ndenumerate(xdata)])

def sum_of_squares(f, xdata, ydata, yerr = None, *args, **kwargs):
    res = residuals(f, xdata, ydata, yerr, *args, **kwargs)
    return np.sum(res**2)

def brute_fit(f, xdata, ydata, yerr = None, bounds = None, Ns = 20,):
    def fun(*args, **kwargs):
        return sum_of_squares(f, xdata, ydata, yerr, *args, **kwargs)
    return brute(fun, ranges = bounds, Ns = Ns, full_output = True)
    

def fit_data(f, xdata, ydata, yerr=None, bounds = None):
    if bounds == None:
        popt, pcov = curve_fit(f, xdata, ydata, sigma=yerr)
    else:
        popt, pcov = curve_fit(f, xdata, ydata, sigma=yerr, bounds = bounds)
    perr = np.sqrt(np.diag(pcov))
    
    # Calculate chi square (reduced)
    if yerr is None: yerr = 1
    residuals = ydata - f(xdata, *popt)
    chisq = np.sum((residuals / yerr) ** 2) / (len(ydata) - len(popt))
    
    return popt, perr, chisq

def main():
    parser_description = "This is a python script for running molscat, collecting and pickling S-matrices, and calculating effective probabilities."
    parser = argparse.ArgumentParser(description=parser_description)
    parser.add_argument("-d", "--phase_step", type = float, default = None, help = "The phase step multiples of pi.")
    parser.add_argument("-s", "--singlet_phase", type = float, default = None, help = "The distinguished value of the singlet semiclassical phase modulo pi in multiples of pi.")
    parser.add_argument("-t", "--triplet_phase", type = float, default = None, help = "The distinguished value of the triplet semiclassical phase modulo pi in multiples of pi.")
    parser.add_argument("--so_scaling", nargs='*', type = float, default = [0.32,], help = "Values of the SO scaling.")
    parser.add_argument("--nenergies", type = int, default = 50, help = "Number of energy values in a grid.")
    parser.add_argument("--E_min", type = float, default = 4e-7, help = "Lowest energy value in the grid.")
    parser.add_argument("--E_max", type = float, default = 4e-3, help = "Highest energy value in the grid.")
    parser.add_argument("--n_grid", type = int, default = 3, help = "n parameter for the nth-root energy grid.")
    parser.add_argument("-T", "--temperatures", nargs='*', type = float, default = None, help = "Temperature in the Maxwell-Boltzmann distributions (in kelvins).")
    parser.add_argument("--nT", type = int, default = 10, help = "Number of temperatures included in the calculations.")
    parser.add_argument("--logT_min", type = float, default = -4)
    parser.add_argument("--logT_max", type = float, default = -3)
    parser.add_argument("--DPhi_input_dir_name", type = str, default = 'RbSr+_fmf_vs_DPhi_SE', help = "Name of the directory with the molscat inputs")
    parser.add_argument("--SO_input_dir_name", type = str, default = 'RbSr+_fmf_so_scaling', help = "Name of the directory with the molscat inputs")
    parser.add_argument("--DPhi", action = 'store_true', help = "if included, DPhi is calculated")
    parser.add_argument("--cso", action = 'store_true', help = "if included, cso is calculated")
    args = parser.parse_args()

    nenergies, E_min, E_max, n = args.nenergies, args.E_min, args.E_max, args.n_grid
    energy_tuple = tuple( round(n_root_scale(i, E_min, E_max, nenergies-1, n = n), sigfigs = 11) for i in range(nenergies) )

    singlet_phase = args.singlet_phase
    singlet_phase_distinguished = singlet_phase if singlet_phase is not None else default_singlet_phase_function(1.0)
    
    so_scaling = args.so_scaling

    phase_step = args.phase_step
    singlet_phases = np.array([default_singlet_phase_function(1.0),]) if phase_step is None else np.arange(phase_step, 1., phase_step).round(decimals=4)
    phase_differences = np.arange(0, 1.+phase_step, phase_step).round(decimals=4)

    temperatures = args.temperatures
    logT_min=args.logT_min
    logT_max=args.logT_max
    nT=args.nT

    if temperatures is None:
        temperatures = list(np.logspace(logT_min, logT_max, nT))
        temperatures.append(5e-4)
        temperatures = np.array(sorted(temperatures))
    else:
        temperatures = np.array(temperatures)

    plot_temperature = 5e-4
    input_dir_name = 'RbSr+_fmf_vs_DPhi_SE'

    if args.DPhi:
        phase_differences, theory_distinguished, experiment, std = get_p0_vs_DPhi(singlet_phases=singlet_phases, phase_differences=phase_differences, so_scaling=0.0, energy_tuple=energy_tuple, singlet_phase_distinguished=singlet_phase_distinguished, temperatures=temperatures, plot_temperature=plot_temperature, input_dir_name=input_dir_name)
        theory_distinguished = np.squeeze(theory_distinguished).transpose()
        theory_hot = theory_distinguished[0]
        theory_cold = theory_distinguished[1]
        popt, perr, chisq = fit_data(spin_exchange, phase_differences, theory_hot, bounds = ((0, 0.5*np.pi), (0.4, 0.7)))
        [Phi0_hot, amplitude_hot] = popt
        print(popt, perr, chisq)
        print(f"{Phi0_hot/np.pi =}")
        popt, perr, chisq = fit_data(spin_exchange, phase_differences, theory_cold, bounds = ((0, 0.5*np.pi), (0.1, 0.4)))
        [Phi0_cold, amplitude_cold] = popt
        print(popt, perr, chisq)
        print(f"{Phi0_cold/np.pi =}")
    
        fig, ax = plt.subplots(figsize = (5, 4.5), dpi = 300)
        
        y1 = spin_exchange(phase_differences, Phi0_hot, amplitude_hot)
        y2 = spin_exchange(phase_differences, Phi0_cold, amplitude_cold)
        ax.plot(phase_differences, y1, color = 'firebrick')
        ax.plot(phase_differences, y1, color = 'midnightblue')

        ax.scatter(phase_differences, theory_hot, s = 16, c = 'firebrick', marker = 'x', edgecolors = 'firebrick')
        ax.scatter(phase_differences, theory_cold, s = 16, c = 'maroon', marker = 'x', edgecolors = 'firebrick')

        ax.set_ylim(0, 0.7)

        fig_path = plots_dir_path / 'trash' / 'fit_sin_vs_DPhi.pdf'
        fig_path.parent.mkdir(parents = True, exist_ok = True)
        plt.savefig(fig_path)
        plt.close()


if __name__ == '__main__':
    main()
