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

from _molscat_data.smatrix import SMatrixCollection
from _molscat_data.thermal_averaging import n_root_scale, n_root_iterator
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

def plotSectionsWithPartialVsTtoFig(fig, phase_step_sections: float, phase_differences: float | np.ndarray[float], phase_difference_distinguished: float, so_scaling: float, energy_tuple: tuple[float, ...], temperatures: tuple[float, ...] = (5e-4,), plot_temperatures = [1e-4, 1e-3, 1e-2], input_dir_name: str = 'RbSr+_tcpld_80mK_0.01_step', transfer_input_dir_name: str = 'RbSr+_tcpld_80mK_0.01_step', hybrid = False, plot_p0 = False, fmf_colormap = False, plot_nan = False,):
    nenergies = len(energy_tuple)
    E_min = min(energy_tuple)
    E_max = max(energy_tuple)
    transfer_nenergies = 200
    # singlet_phases, triplet_phases = np.array(singlet_phases), np.array(triplet_phases)
    probabilities_dir_name = 'probabilities_hybrid' if hybrid else 'probabilities'
 
    F1, F2 = 2, 1
    MF1, MF2 = -2, 1

    pickle_path = pickles_dir_path / input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{0.04:.4f}_{0.04+phase_difference_distinguished:.4f}' / f'{so_scaling:.4f}' / f'in_{F1}_{MF1}_{F2}_{MF2}.pickle'
    transfer_pickle_path = pickles_dir_path / transfer_input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{transfer_nenergies}_E' / f'{0.04:.4f}_{0.04+phase_difference_distinguished:.4f}' / f'{0.0:.4f}' / f'in_4_4_1_1.pickle'

    s_matrix_collection = SMatrixCollection.fromPickle(pickle_path)
    l_max = int(max(key[0].L for s_matrix in s_matrix_collection.matrixCollection.values() for key in s_matrix.matrix.keys())/2)

    transfer_s_matrix_collection = SMatrixCollection.fromPickle(transfer_pickle_path)
    transfer_l_max = int(max(key[0].L for s_matrix in transfer_s_matrix_collection.matrixCollection.values() for key in s_matrix.matrix.keys())/2)   

    exp_cold_lower = np.loadtxt(data_dir_path / 'exp_data' / ('p0_single_ion_cold_lower.dat' if plot_p0 else 'single_ion_cold_lower.dat'))

    experiment = np.array([exp_cold_lower[0,0],])
    std = np.array([exp_cold_lower[1,0],])

    gs = gridspec.GridSpec(len(plot_temperatures),1, fig)
    gs.update(hspace=0.0)
    print(f'{plot_temperatures = }')
    fig_axs = [fig.add_subplot(gs[i,:]) for i in range(len(plot_temperatures))]
    [ax.sharex(fig_axs[0]) for ax in fig_axs[1:]]

    # T_index = np.nonzero(temperatures == plot_temperature)[0][0]

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

            k_archive_paths = [arrays_dir_path / input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{(singlet_phase+phase_difference_distinguished)%1:.4f}' / f'{so_scaling:.4f}.zip' if (singlet_phase+phase_difference_distinguished)%1 != 0 else None for singlet_phase in singlet_phases_sections]
            k_L_E_array_paths = [[arrays_dir_path / input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{(singlet_phase+phase_difference_distinguished)%1:.4f}' / f'{so_scaling:.4f}' / f'in_{F1}_{MF1}_{F2}_{MF2}' / 'k_L_E'/ 'cold_lower' / f'OUT_{F1}_{MF_out}_{F2}_{MF2-2}_IN_{F1}_{MF1}_{F2}_{MF2}.txt' if (singlet_phase+phase_difference_distinguished)%1 != 0 else None for MF_out in range(-F1, F1+1, 2)] for singlet_phase in singlet_phases_sections]
            k_m_L_E_array_paths = [[arrays_dir_path / input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{(singlet_phase+phase_difference_distinguished)%1:.4f}' / f'{so_scaling:.4f}' / f'in_{F1}_{MF1}_{F2}_{MF2}' / 'k_m_L_E'/ 'cold_lower' / f'OUT_{F1}_{MF_out}_{F2}_{MF2-2}_IN_{F1}_{MF1}_{F2}_{MF2}.txt' if (singlet_phase+phase_difference_distinguished)%1 != 0 else None for MF_out in range(-F1, F1+1, 2)] for singlet_phase in singlet_phases_sections]

            print("Starting extracting!")
            for archive_path in k_archive_paths:
                print(archive_path)
                if archive_path is None: continue
                with zipfile.ZipFile(archive_path, 'r') as zObject:
                    for MF_out in range(-F1, F1+1, 2):
                        if not (archive_path.with_suffix('') / Path(f'in_{F1}_{MF1}_{F2}_{MF2}/k_L_E/cold_lower/OUT_{F1}_{MF_out}_{F2}_{MF2-2}_IN_{F1}_{MF1}_{F2}_{MF2}.txt')).is_file():
                            zObject.extract(f'in_{F1}_{MF1}_{F2}_{MF2}' / 'k_L_E'/ 'cold_lower' / f'OUT_{F1}_{MF_out}_{F2}_{MF2-2}_IN_{F1}_{MF1}_{F2}_{MF2}.txt', archive_path.with_suffix(''))
                        if not (archive_path.with_suffix('') / Path(f'in_{F1}_{MF1}_{F2}_{MF2}/k_m_L_E/cold_lower/OUT_{F1}_{MF_out}_{F2}_{MF2-2}_IN_{F1}_{MF1}_{F2}_{MF2}.txt')).is_file():
                            zObject.extract(f'in_{F1}_{MF1}_{F2}_{MF2}' / 'k_m_L_E'/ 'cold_lower' / f'OUT_{F1}_{MF_out}_{F2}_{MF2-2}_IN_{F1}_{MF1}_{F2}_{MF2}.txt', archive_path.with_suffix(''))
            print("Finished extracting!")

        else:
            array_paths_cold_lower_distinguished = [arrays_dir_path / input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{(singlet_phase+phase_difference_distinguished)%1:.4f}' / f'{so_scaling:.4f}' / probabilities_dir_name / ('p0_cold_lower.txt' if plot_p0 else 'cold_lower.txt') if ( singlet_phase + phase_difference_distinguished ) % 1 !=0 else None for singlet_phase in singlet_phases_sections]
            arrays_cold_lower_distinguished = np.array([ np.loadtxt(array_path) if (array_path is not None and array_path.is_file()) else np.full((len(temperatures), 3), np.nan) for array_path in array_paths_cold_lower_distinguished ])
            arrays_cold_lower_distinguished = arrays_cold_lower_distinguished.reshape(arrays_cold_lower_distinguished.shape[0], len(temperatures), -1)

    if phase_difference_distinguished is not None and fmf_colormap:
        print("Starting loading k_L_E arrays")
        k_L_E_arrays = np.array([ [np.loadtxt(array_path_MF_out) if (array_path_MF_out is not None and array_path_MF_out.is_file()) else np.full((l_max+1,50), np.nan) for array_path_MF_out in array_for_singlet_phase] for array_for_singlet_phase in k_L_E_array_paths])
        print("Starting loading k_m_L_E arrays")
        k_m_L_E_arrays = np.array([ [np.loadtxt(array_path_MF_out) if (array_path_MF_out is not None and array_path_MF_out.is_file()) else np.full((transfer_l_max,transfer_nenergies), np.nan) for array_path_MF_out in array_for_singlet_phase] for array_for_singlet_phase in k_m_L_E_array_paths ])
        print("Finished loading the arrays")

        distribution_arrays = [np.fromiter(n_root_iterator(temperature = temperature, E_min = min(s_matrix_collection.collisionEnergy), E_max = max(s_matrix_collection.collisionEnergy), N = len(s_matrix_collection.collisionEnergy), n = 3), dtype = float) for temperature in temperatures]
        transfer_distribution_arrays = [np.fromiter(n_root_iterator(temperature = temperature, E_min = min(transfer_s_matrix_collection.collisionEnergy), E_max = max(transfer_s_matrix_collection.collisionEnergy), N = len(transfer_s_matrix_collection.collisionEnergy), n = 3), dtype = float) for temperature in temperatures]
        
        print("Starting calculating energy averages")
        average_rate_arrays = np.array( [s_matrix_collection.thermalAverage(k_L_E_arrays, distribution_array) for distribution_array in distribution_arrays ] )
        #### summing over all values of MF_out
        ## average_rate_arrays = np.sum(average_rate_arrays, axis = 1)
        average_rate_arrays = np.moveaxis(average_rate_arrays, -1, 0)
        ### now we have (L, T, singlet_phase, MF_out) indices on axes for average_rate_arrays and expected shape (50, 21, 98, 3)
        average_momentum_transfer_arrays = np.array( [ transfer_s_matrix_collection.thermalAverage(k_m_L_E_arrays.sum(axis=-2), transfer_distribution_array) for transfer_distribution_array in transfer_distribution_arrays ] )
        ### now we have (T, singlet_phase, MF_out) indices on axes for average_momentum_transfer_arrays and expected shape (21, 98, 3)
        print("Finished calculating energy averages")
        print(f'{average_rate_arrays.shape = }, {average_momentum_transfer_arrays.shape = }')

        probability_arrays = average_rate_arrays / average_momentum_transfer_arrays
        ### now we the have (L, T, singlet_phase, MF_out) indices on axes for probability_arrays and expected shape (50, 21, 98, 3)
        ### we sum over all values of MF_out and move L-axis to the last position
        probability_arrays = np.moveaxis(probability_arrays.sum(axis=-1),0,-1)
        ### now we the have (T, singlet_phase, L) indices on axes for probability_arrays and expected shape (21, 98, 50)
        print(f'{probability_arrays.shape = }')

    fig1_ax0 = fig.add_subplot()
    fig1_ax1 = fig.add_subplot(sharex = fig1_ax0)

    ### Plot sections for a single temperature but a few values of the phase difference

    # T_index = np.nonzero(temperatures == plot_temperature)[0][0]
    T_indices = np.array([np.abs(temperatures - value).argmin() for value in plot_temperatures])
    print(f'{T_indices = }')
    # theory = arrays_cold_lower[:,:,T_index,0]
    # print(f'{theory = }')
    # if plot_nan:
    #     theory[np.isnan(theory)] = (theory[np.roll(np.isnan(theory),-1,0)]+theory[np.roll(np.isnan(theory),1,0)])/2
    # theory_distinguished = np.moveaxis(np.array( [ arrays_cold_lower_distinguished[:,T_index, 0], ]), 0, -1)

    # theory_vs_Phis = theory
    # theory_vs_Phis_distinguished = theory_distinguished
 
 
    ### Plot sections for the fitted value of the phase difference but many temperatures
    
    color_map = cmocean.cm.thermal
    lognorm = matplotlib.colors.LogNorm(vmin=min(temperatures), vmax=max(temperatures), clip = False)
    theory_colors = [color_map(lognorm(temperature)) for temperature in temperatures[::2]]
    L_color_map = matplotlib.colormaps['inferno']
    plot_l_max = 20
    L_norm = matplotlib.colors.Normalize(vmin=0, vmax=plot_l_max, clip = False)
    if phase_difference_distinguished is not None and fmf_colormap:
        theory_formattings = [ *[{'color': color, 'linewidth': 1.25} for color in theory_colors],
                              *[{'color': L_color_map(L_norm(L)), 'linewidth': 0.25} for L in range(0,plot_l_max+1)]
                               ]
    else:
        theory_formattings = [ {'color': color, 'linewidth': 1.25} for color in theory_colors ]
    # theory_distinguished_formattings = [ {'color': 'k', 'linewidth': 4, 'linestyle':  (1.05,(0.1,2)), 'dash_capstyle': 'round' } for exp in experiment]
    # mrkcolor='#b50033' # ładny optymalny czerwony # '#f5390a'
    # mrkcolor = '#ff1414ff'# jaskrawy czerwony
    mrkcolor = '#cc0000ff' # czerwony jak atom rubidu
    theory_distinguished_formattings = [ {'color': 'k', 'linewidth': 0,
                                          'markevery': 0.03, 'markersize': 3,
                                          'marker': 'o', 'markeredgecolor': mrkcolor, 'markerfacecolor': mrkcolor} for exp in experiment]

    # T_index = np.nonzero(temperatures == plot_temperature)[0][0]
    T_indices = np.array([np.abs(temperatures - value).argmin() for value in plot_temperatures])
    print(f'{T_indices = }')
    # print(f'{arrays_cold_lower_distinguished =}')
    # print(f'{arrays_cold_lower_distinguished.shape =}')

    # print(f'{arrays_cold_lower_distinguished[:,::2,0].shape =}')
    # print(f'{arrays_cold_lower_distinguished[:,T_indices,0].shape = }')
    print(f'{np.transpose(arrays_cold_lower_distinguished[:,T_indices,0]).reshape(len(T_indices), -1, 1).shape = }')
    print(f'{np.transpose(probability_arrays[T_indices,:,:(plot_l_max+1)], (0,2,1)).shape =}')
    # print(f'{probability_arrays[T_index,:,:].shape = }')
    ### now we the have (T, singlet_phase, L) indices on axes for probability_arrays and expected shape (21, 98, 50)
    if phase_difference_distinguished is not None and fmf_colormap:
        theory = np.array([
            np.transpose([
                arrays_cold_lower_distinguished[:,index,0],
                *np.transpose(probability_arrays[index,:,:(plot_l_max+1)])
            ])
            for index in T_indices
        ])
        # theory = np.transpose([*np.transpose(arrays_cold_lower_distinguished[:,T_indices,0]).reshape(len(T_indices), 1, -1), 
        #                        *np.transpose(probability_arrays[T_indices,:,:(plot_l_max+1)], (0,2,1))],
        #           (0,2,1)
        #           )
    ### powinien być na końcu kształt (T, singlet_phase, plot_l_max+2), czyli (3, 98, 21), bo dla każdej temperatury i Phis mamy plot_l_max+1 fal parcjalnych i jedno sumaryczne prawdopobieństwo
    else:
        theory = np.moveaxis(arrays_cold_lower_distinguished[:,::2,0], 1, -1)
    
    # theory = np.moveaxis(arrays_cold_lower_distinguished[:,::2,0], 1, -1)
    
    print(f'{theory.shape = }')
    print(f'{theory = }')

    if plot_nan:
        theory[np.isnan(theory)] = (theory[np.roll(np.isnan(theory),-1,0)]+theory[np.roll(np.isnan(theory),1,0)])/2
    theory_distinguished = np.moveaxis(np.array( [ arrays_cold_lower_distinguished[:,T_indices, 0], ]), 0, 1)
    # print(f'{theory = }')


    # sections_temperatures = temperatures[::2]
    theory_vs_T, theory_vs_T_distinguished = theory, theory_distinguished

    for index, ax in enumerate(fig_axs):
        print(f'{singlet_phases_sections.shape = }')
        print(f'{theory[index].shape = }')
        print(f'{theory_distinguished[index].shape =}')
        print(f'{theory_formattings.shape = }')
        print(f'{theory_distinguished_formattings.shape = }')
        ax = ValuesVsModelParameters.plotValuestoAxis(ax, singlet_phases_sections, theory[index], experiment, std, theory_distinguished[index], theory_formattings, theory_distinguished_formattings)
        PhaseTicks.linearStr(ax.yaxis, 0.1 if plot_p0 else 0.2, 0.05 if plot_p0 else 0.1, '${x:.1f}$')
        ax.set_ylim(0, ax.get_ylim()[1])

        filter_max_probability = np.equal(np.full_like(probability_arrays[index,:,:(plot_l_max+1)], np.nanmax(probability_arrays[index,:,:(plot_l_max+1)], axis = 0)).transpose(), probability_arrays[index,:,:(plot_l_max+1)].transpose())
        print(f'{filter_max_probability.shape = }')
        print(f'{filter_max_probability = }')
        # print(f'{filter_max_probability == True}')
        ##### find the maximum for each partial wave and return tuples of the form (L, Phis_max, k_max)
        coords_vs_L = tuple( (l, singlet_phases_sections[filter_max_probability[l]], probability_arrays[index,:,l][filter_max_probability[l]]) for l in range(plot_l_max+1) if np.any(filter_max_probability[l]) and np.any(probability_arrays[index,:,l][filter_max_probability[l]] > 0.10*np.nanmax(probability_arrays[index,:,:].sum(axis=1))) )
        print(f'{coords_vs_L = }')

        # annotate peaks with the orbital quantum numbers L
        for coord in coords_vs_L:
            ax.text(coord[1], coord[2] + (ax.get_ylim()[1]-ax.get_ylim()[0])*0.02, f'{coord[0]}', fontsize = 'x-small', color = L_color_map(L_norm(coord[0])), va = 'center', ha = 'center')#fontweight = 'bold', 

        # set y-label
        ax.set_ylabel(f'$p_0$' if plot_p0 else f'$p_\\mathrm{{eff}}$')#, rotation = 0, lapelpad = 12)

    ### Set the x-ticks and x-label
    PhaseTicks.setInMultiplesOfPhi(fig_axs[0].xaxis)
    fig_axs[-1].set_xlabel(f'$\\Phi_\\mathrm{{s}}$')

    ### Set the grid so that the scale on both subplots is not deformed
    # lim0 = fig1_ax0.get_ylim()
    # lim1 = fig1_ax1.get_ylim()
    # print(f'{lim1 = }')

    return fig, fig_axs, gs, singlet_phases_sections

def plotMagneticFieldtoFig(fig, magnetic_phases: tuple[tuple[float, float], ...], magnetic_fields: float | np.ndarray[float], magnetic_field_experimental: float, energy_tuple: tuple[float, ...], temperatures: tuple[float, ...] = (5e-4,), plot_temperatures: float = [1e-4, 1e-3, 1e-2], input_dir_name: str = 'RbSr+_fmf_SE_vs_B_80mK', plot_p0 = False, so_scaling = None,):
    ## (c) Spin-exchange probabilities vs the magnetic field
    nenergies = len(energy_tuple)
    E_min = min(energy_tuple)
    E_max = max(energy_tuple)
    probabilities_dir_name = 'probabilities'
    prefix_for_array_path = '' if not plot_p0 else 'p0_'
    F1, F2 = 2, 1
    MF1, MF2 = -2, 1

    gs = gridspec.GridSpec(3,1, fig)
    gs.update(hspace=0.0)
    fig_axs = [fig.add_subplot(gs[i,:]) for i in range(len(plot_temperatures))]
    [ax.sharex(fig_axs[0]) for ax in fig_axs[1:]]
    # [ax.sharey(fig_axs[0]) for ax in fig_axs[1:]]
    
    if so_scaling is None:
        abbreviation='cold'
        array_paths_cold_lower = [  [arrays_dir_path / input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{triplet_phase:.4f}' / f'{F1}_{MF1}_{F2}_{MF2}' / f'{magnetic_field:.2f}' / probabilities_dir_name / f'{prefix_for_array_path}{abbreviation}.txt' for magnetic_field in magnetic_fields] for singlet_phase, triplet_phase in magnetic_phases]
        [ [print(array_path) for array_path in sublist if (array_path is not None and not array_path.is_file())] for sublist in array_paths_cold_lower ]
        arrays_cold_lower = np.array([ [np.loadtxt(array_path) if (array_path is not None and array_path.is_file()) else np.full((len(temperatures), 3), np.nan) for array_path in sublist] for sublist in array_paths_cold_lower ])
        arrays_cold_lower = arrays_cold_lower.reshape(*arrays_cold_lower.shape[:2], len(temperatures), -1)

    else:
        abbreviation='cold_lower'
        array_paths_cold_lower = [  [arrays_dir_path / input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{triplet_phase:.4f}' / f'{so_scaling:.4f}' / f'in_{F1}_{MF1}_{F2}_{MF2}' / f'{magnetic_field:.2f}' / probabilities_dir_name / f'{prefix_for_array_path}{abbreviation}.txt' for magnetic_field in magnetic_fields] for singlet_phase, triplet_phase in magnetic_phases]
        [ [print(array_path) for array_path in sublist if (array_path is not None and not array_path.is_file())] for sublist in array_paths_cold_lower ]
        arrays_cold_lower = np.array([ [np.loadtxt(array_path) if (array_path is not None and array_path.is_file()) else np.full((len(temperatures), 3), np.nan) for array_path in sublist] for sublist in array_paths_cold_lower ])
        arrays_cold_lower = arrays_cold_lower.reshape(*arrays_cold_lower.shape[:2], len(temperatures), -1)

    exp_cold_lower = np.loadtxt(data_dir_path / 'exp_data' / ('p0_single_ion_cold_lower.dat' if plot_p0 else 'single_ion_cold_lower.dat'))

    experiment = np.array([exp_cold_lower[0,0],])
    std = np.array([exp_cold_lower[1,0],])

    T_indices = np.array([np.abs(temperatures - value).argmin() for value in plot_temperatures])
    print(f'{T_indices = }')
    # T_index = np.nonzero(temperatures == plot_temperature)[0][0]
    ### indices: (phases, magnetic_field, temperature)
    theory = np.transpose( arrays_cold_lower[:,:,T_indices,0], (2,1,0))
    ### indices now: (temperature, magnetic_field, phases)
    theory_distinguished = None
    print(f'{theory.shape = }')

    theory_vs_B = theory

    color_map = cmcrameri.cm.devon
    theory_colors = list(reversed([color_map(singlet_phase) for singlet_phase, triplet_phase in magnetic_phases]))
    theory_formattings = [ {'color': color, 'linewidth': 1.25} for color in theory_colors ]
    # theory_distinguished_formattings = [ {'color': 'k', 'linewidth': 4, 'linestyle':  (1.05,(0.1,2)), 'dash_capstyle': 'round' } for exp in experiment]
    # mrkcolor = '#b50033' # ładny optymalny czerwony # '#f5390a'
    # mrkcolor = '#ff1414ff'# jaskrawy czerwony
    mrkcolor = '#cc0000ff' # czerwony jak atom rubidu
    theory_distinguished_formattings = [ {'color': 'k', 'linewidth': 0,
                                          'markevery': 0.06, 'markersize': 3,
                                          'marker': 'o', 'markeredgecolor': mrkcolor, 'markerfacecolor': mrkcolor} for exp in experiment]

    # fig2_ax = fig.add_subplot()

    # print(f'{theory = }')
    for index, ax in enumerate(fig_axs):
        ax = ValuesVsModelParameters.plotValuestoAxis(ax, magnetic_fields, theory[index], experiment=None, std=None, theory_distinguished=None, theory_formattings = theory_formattings, theory_distinguished_formattings=theory_distinguished_formattings)
        # fig2_ax.scatter([magnetic_field_experimental,], experiment, s = 16, c = theory_distinguished_formattings[0]['color'], marker = 'd', edgecolors = 'dodgerblue')
        # fig2_ax.errorbar([magnetic_field_experimental, ], experiment, std, ecolor = theory_distinguished_formattings[0]['color'], capsize = 6)
        ax.set_ylim(0, 1.05*ax.get_ylim()[1])
        PhaseTicks.linearStr(ax.yaxis, 0.1, 0.05, '${x:.1f}$')
        for i, (singlet_phase, triplet_phase) in enumerate(magnetic_phases):
            ax.get_lines()[i].set_label(f'$\\Phi_\\mathrm{{s}} = {singlet_phase:.2f}\\pi$')
        # labelLines(fig2_ax.get_lines(), align = False, outline_width=2, color = 'white', fontsize = matplotlib.rcParams["xtick.labelsize"], )
        labelLines(ax.get_lines(), align = False, outline_color = None, yoffsets= -6.7e-3*(ax.get_ylim()[1]-ax.get_ylim()[0]), fontsize = matplotlib.rcParams["xtick.labelsize"], )
        props = dict(boxstyle='round', facecolor='none', edgecolor='none')
        # props = dict(facecolor='none')
        # ax.text(0.03, 0.10, f'$\\Delta\\Phi_\\mathrm{{fit}} = {(magnetic_phases[0][1]-magnetic_phases[0][0])%1:.2f}\\pi$', va = 'center', ha = 'left', transform = ax.transAxes, bbox = props)
        ax.text(0.03, 0.10, f'$\\Delta\\Phi_\\mathrm{{fit}} = {plot_temperatures[index]:.2e}\\,\\mathrm{{K}}$', va = 'center', ha = 'left', transform = ax.transAxes, bbox = props)
        ylabel = f'$p_\\mathrm{{eff}}$' if not plot_p0 else f'$p_0$'
        ax.set_ylabel(ylabel)
    
    PhaseTicks.linearStr(fig_axs[0].xaxis, 100, 20, '${x:n}$') if max(magnetic_fields)-min(magnetic_fields) > 250 else PhaseTicks.linearStr(fig_axs[0].xaxis, 50, 10, '${x:n}$')
    fig_axs[-1].set_xlabel(f'$B\\,(\\mathrm{{G}})$')

    for ax in fig_axs[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)

    for ax in fig_axs[1:]:
        ax.yaxis.get_major_ticks()[-1].label1.set_visible(False)
        # ax.yaxis.get_major_ticks()[-2].label1.set_visible(False)


    return fig, fig_axs, gs, theory_vs_B,

def plotSupplementaryFig2(phase_step_cm: float, phase_step_sections: float, phase_differences: float | np.ndarray[float], phase_difference_distinguished: float, so_scaling: float, magnetic_phases: tuple[tuple[float, float], ...], magnetic_fields: float | np.ndarray[float], magnetic_field_experimental: float, MF_in: int, MS_in: int, energy_tuple: tuple[float, ...], temperatures: tuple[float, ...] = (5e-4,), plot_temperatures: float = [1e-4, 1e-3, 1e-2], cm_input_dir_name: str = 'RbSr+_tcpld_80mK_0.01_step', vs_B_input_dir_name = 'RbSr+_fmf_vs_SE_80mK', cm_transfer_input_dir_name = 'RbSr+_tcpld_80mK_0.01_step', colormap_hybrid = False, plot_p0 = False, plot_section_lines = False, fmf_colormap = False, so_scaling_vs_B = False, plot_nan = False, journal_name = 'NatCommun'):
    plt.style.use(Path(__file__).parent / 'mpl_style_sheets' / f'{journal_name}.mplstyle')
    nenergies = len(energy_tuple)
    E_min = min(energy_tuple)
    E_max = max(energy_tuple)
    suffix = '_hybrid' if colormap_hybrid else ''
    png_path = plots_dir_path / 'paper' / f'{journal_name}' / 'SupplementaryFig2' / f'{cm_input_dir_name}_{vs_B_input_dir_name}' / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'SupplementaryFig2.png'
    pdf_path = png_path.with_suffix('.pdf')
    svg_path = png_path.with_suffix('.svg')
    data_path = png_path.with_suffix('.txt')
    png_path.parent.mkdir(parents = True, exist_ok = True)

    cm = 1/2.54
    ws, hs = 0.05, 0.05
    total_height = 7.5
    figsize = (18*cm, total_height*cm)
    dpi = 1200
    fig = plt.figure(figsize=figsize, dpi = dpi)
    gs_Figure = gridspec.GridSpec(1,180, fig)
    # figs = fig.subfigures(2, 2, wspace = ws, hspace = hs)
    fig0 = fig.add_subfigure(gs_Figure[:,:90])
    fig1 = fig.add_subfigure(gs_Figure[:,90:])

    fig0, fig0_axs, gs0, singlet_phases_sections = plotSectionsWithPartialVsTtoFig(fig = fig0, phase_step_sections = phase_step_sections, phase_differences = phase_differences, phase_difference_distinguished = phase_difference_distinguished, so_scaling = so_scaling, energy_tuple = energy_tuple, temperatures = temperatures, plot_temperatures = plot_temperatures, input_dir_name = cm_input_dir_name, transfer_input_dir_name = cm_transfer_input_dir_name, hybrid = colormap_hybrid, plot_p0 = plot_p0, fmf_colormap = fmf_colormap, plot_nan = plot_nan)
 
    ###### Save data from figures to .txt files
    # np.savetxt(data_path.with_stem(data_path.stem+'_colormap_singlet_phases'), _singlet_phases_cm, fmt = '%.4f')
    # np.savetxt(data_path.with_stem(data_path.stem+'_colormap_triplet_phases'), _triplet_phases_cm, fmt = '%.4f')
    # np.savetxt(data_path.with_stem(data_path.stem+'_colormap_theory'), _theory_cm, fmt = '%.4f')
    # np.savetxt(data_path.with_stem(data_path.stem+'_sections_singlet_phases'), _singlet_phases_sections, fmt = '%.4f')
    # np.savetxt(data_path.with_stem(data_path.stem+'_sections_DPhi'), phase_differences, fmt = '%.4f')
    # np.savetxt(data_path.with_stem(data_path.stem+'_sections_theory_vs_Phis'), _theory_vs_Phis, fmt = '%.4f')
    # np.savetxt(data_path.with_stem(data_path.stem+'_sections_theory_vs_Phis_distiguished'), _theory_vs_Phis_distinguished, fmt = '%.4f')
    # np.savetxt(data_path.with_stem(data_path.stem+'_sections_theory_vs_T'), _theory_vs_T, fmt = '%.4f')
    # np.savetxt(data_path.with_stem(data_path.stem+'_sections_theory_vs_T_distinguished'), _theory_vs_T_distinguished, fmt = '%.4f')
    # np.savetxt(data_path.with_stem(data_path.stem+'_sections_temperatures'), _sections_temperatures, fmt = '%.4e')

    fig1, fig1_axs, gs1, theory_vs_B, = plotMagneticFieldtoFig(fig1, magnetic_phases, magnetic_fields, magnetic_field_experimental, energy_tuple, temperatures, plot_temperatures, vs_B_input_dir_name, plot_p0 = plot_p0, so_scaling = (so_scaling if so_scaling_vs_B else None))
    # fig2, fig2_ax, fig3, fig3_axs, gs3, _theory_vs_B, _vs_B_temperatures, _theory_vs_B_vs_T, _theory_vs_B_vs_T_distinguished = plotMagneticFieldtoFigs(fig2, fig3, magnetic_phases, magnetic_fields, magnetic_field_experimental, energy_tuple, temperatures, plot_temperature, vs_B_input_dir_name, plot_p0 = plot_p0, so_scaling = (so_scaling if so_scaling_vs_B else None))

    ###### Save data from Fig3d-e to .txt files
    # np.savetxt(data_path.with_stem(data_path.stem+'_vs_B_magnetic_fields'), magnetic_fields, fmt = '%.4f')
    # with open(data_path.with_stem(data_path.stem+'_vs_B_phases'), 'w') as f:
    #     print(magnetic_phases, file = f)
    # np.savetxt(data_path.with_stem(data_path.stem+'_vs_B_theory'), _theory_vs_B, fmt = '%.4f')
    # for i in range(_theory_vs_B_vs_T.shape[0]):
    #     np.savetxt(data_path.with_stem(data_path.stem+f'_vs_B_vs_T_theory_{i}'), _theory_vs_B_vs_T[i], fmt = '%.4f')
    #     np.savetxt(data_path.with_stem(data_path.stem+f'_vs_B_vs_T_theory_distinguished_{i}'), _theory_vs_B_vs_T_distinguished[:,i], fmt = '%.4f')
    # np.savetxt(data_path.with_stem(data_path.stem+'_vs_B_temperatures'), _vs_B_temperatures, fmt = '%.4e')


    # fig0_ax.text(0., 1.0, f'a', fontsize = 8, family = 'sans-serif', va = 'top', ha = 'left', transform = fig.transFigure, fontweight = 'bold')
    fig1_axs[0].text(0.52, 1.00, f'b', fontsize = 8, family = 'sans-serif', va = 'top', ha = 'left', transform = fig.transFigure, fontweight = 'bold')

    fig0.subplots_adjust(left = 0.05)
    gs1.update(left = 0.17, right = 0.97)
    # fig1.subplots_adjust(left = 0.17, right = 0.97)
    # fig2.subplots_adjust(left = 0.1, right = 0.97)
    # fig3.subplots_adjust(left = 0.17, right = 1-(0.03)*90/60)
    # gs3.update(left = 0.17, right = 1-(0.03)*90/60)

    fig.savefig(png_path, bbox_inches='tight', pad_inches = 0)
    fig.savefig(svg_path, bbox_inches='tight', pad_inches = 0, transparent = True)
    fig.savefig(pdf_path, bbox_inches='tight', pad_inches = 0, transparent = True)

    plt.close()

def main():
    parser_description = "This is a python script for running molscat, collecting and pickling S-matrices, and calculating effective probabilities."
    parser = argparse.ArgumentParser(description=parser_description)
    parser.add_argument("--phase_step_cm", type = float, default = 0.04, help = "The phase step in the multiples of pi for the color map.")
    parser.add_argument("--phase_step_sections", type = float, default = 0.01, help = "The singlet phase step in themultiples of pi for the sections through the color map.")
    parser.add_argument("--phase_differences", nargs='*', type = float, default = None, help = "The values of the singlet-triplet semiclassical phase difference modulo pi in multiples of pi.")
    parser.add_argument("--phase_difference", type = float, default = 0.2, help = "The distinguished value of the singlet-triplet semiclassical phase difference modulo pi in multiples of pi.")
    parser.add_argument("--so_scaling", type = float, default = 0.32, help = "Value of the SO scaling.")

    parser.add_argument("-s", "--singlet_phases", nargs='*', type = float, default = [0.04,], help = "The singlet semiclassical phase modulo pi in multiples of pi for the plot of magnetic field.")
    parser.add_argument("-t", "--triplet_phases", nargs='*', type = float, default = [0.24,], help = "The triplet semiclassical phase modulo pi in multiples of pi for the plot of magnetic field.")

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
    parser.add_argument("--plot_temperatures", nargs='*', type = float, default = [1e-4, 1.1e-3, 1e-2])

    parser.add_argument("--fmf_colormap", action = 'store_true', help = "Assume that the scattering calculations in molscat were done with the fmf basis set. Changes the directory structure for arrays.")
    parser.add_argument("--so_scaling_vs_B", action = 'store_true', help = "Assume that the scattering calculations in molscat were done with spin-orbit coupling included. Changes the directory structure for arrays.")

    parser.add_argument("--cm_input_dir_name", type = str, default = 'RbSr+_fmf_so_scaling', help = "Name of the directory with the molscat inputs")
    parser.add_argument("--vs_B_input_dir_name", type = str, default = 'RbSr+_fmf_so_scaling', help = "Name of the directory with the molscat inputs")
    parser.add_argument("--cm_transfer_input_dir_name", type = str, default = 'RbSr+_fmf_momentum_transfer', help = "Name of the directory with the molscat inputs")
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

    print("FUCK YOU")
    plotSupplementaryFig2(phase_step_cm = args.phase_step_cm, phase_step_sections = args.phase_step_sections, phase_differences = phase_differences, phase_difference_distinguished = args.phase_difference, so_scaling = so_scaling, magnetic_phases = magnetic_phases, magnetic_fields = magnetic_fields, magnetic_field_experimental = 2.97, MF_in = MF1, MS_in = MF2, energy_tuple = energy_tuple, temperatures = temperatures, plot_temperatures = args.plot_temperatures, cm_input_dir_name = args.cm_input_dir_name, vs_B_input_dir_name = args.vs_B_input_dir_name, cm_transfer_input_dir_name = args.cm_transfer_input_dir_name, colormap_hybrid = args.colormap_hybrid, plot_p0 = args.plot_p0, plot_section_lines = args.plot_section_lines, journal_name = args.journal, fmf_colormap = args.fmf_colormap, so_scaling_vs_B = args.so_scaling_vs_B, plot_nan = args.plot_nan)

if __name__ == '__main__':
    main()