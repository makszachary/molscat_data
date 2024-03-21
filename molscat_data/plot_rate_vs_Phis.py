import sys
import os
from pathlib import Path
import shutil
from zipfile import ZipFile
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




def plotRateVsPhisForEachEnergy(phase_step: float, phase_difference: float, so_scaling: float, energy_tuple: tuple[float, ...], input_dir_name: str = 'RbSr+_fmf_so_scaling', plot_nan = False, journal_name = 'NatCommun'):
    plt.style.use(Path(__file__).parent / 'mpl_style_sheets' / f'{journal_name}.mplstyle')

    time_0 = time.perf_counter()
    cm = 1/2.54
    total_height=8
    figsize = (18*cm, total_height*cm)
    dpi = 1200

    molscat_out_dir = scratch_path.joinpath('molscat', 'outputs')

    nenergies = len(energy_tuple)
    E_min = min(energy_tuple)
    E_max = max(energy_tuple)
    # singlet_phases, triplet_phases = np.array(singlet_phases), np.array(triplet_phases)

    F1, F2 = 2, 1
    MF1, MF2 = -2, 1
    MF1out, MF2out = 0, -1

    ## SECTIONS THROUGH THE CONTOUR MAP

    singlet_phases = np.array([default_singlet_phase_function(1.0),]) if phase_step is None else np.arange(phase_step, 1., phase_step).round(decimals=4)

    zipped_dir_paths = [ arrays_dir_path / input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{(singlet_phase+phase_difference)%1:.4f}' / f'{so_scaling:.4f}' for singlet_phase in singlet_phases if ( singlet_phase+phase_difference ) % 1 !=0]
    for zipped_dir_path in zipped_dir_paths:
        zip_path = zipped_dir_path.parent / (zipped_dir_path.name + '.zip')
        if zip_path.is_file():        
            shutil.unpack_archive(zip_path, zipped_dir_path, 'zip')

    k_L_E_array_paths = [  arrays_dir_path / input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{(singlet_phase+phase_difference)%1:.4f}' / f'{so_scaling:.4f}' / f'in_{F1}_{MF1}_{F2}_{MF2}' / 'k_L_E' / 'cold_lower' / f'OUT_{F1}_{MF1out}_{F2}_{MF2out}_IN_{F1}_{MF1}_{F2}_{MF2}.txt' if ( singlet_phase+phase_difference ) % 1 !=0 else None for singlet_phase in singlet_phases]
    [print(array_path) for array_path in k_L_E_array_paths if (array_path is not None and not array_path.is_file())]
    print(k_L_E_array_paths)
    k_L_E_arrays = np.array([np.loadtxt(array_path) if (array_path is not None and array_path.is_file()) else np.full((50, nenergies), np.nan) for array_path in k_L_E_array_paths]).transpose(1,2,0)
    if plot_nan:
        k_L_E_arrays[np.isnan(k_L_E_arrays)] = (k_L_E_arrays[np.roll(np.isnan(k_L_E_arrays),-1,0)]+k_L_E_arrays[np.roll(np.isnan(k_L_E_arrays),1,0)])/2
    total_k_E_Phis_array = k_L_E_arrays.sum(axis = 0)

    filter_max_arr = np.equal(np.full_like(k_L_E_arrays.transpose(2,0,1), np.nanmax(k_L_E_arrays, axis = 2)).transpose(1,2,0), k_L_E_arrays)
    # print(f'{k_L_E_arrays.transpose(2,0,1) = }')
    # print(f'{np.amax(k_L_E_arrays, axis = 2) = }')
    # print(f'{np.full_like(k_L_E_arrays.transpose(2,0,1), np.amax(k_L_E_arrays, axis = 2)) = }')
    # print(f'{np.full_like(k_L_E_arrays.transpose(2,0,1), np.amax(k_L_E_arrays, axis = 2)).transpose(1,2,0) = }')
    # print(f'{k_L_E_arrays = }')
    # print(f'{filter_max_arr = }')
    # return
    color_map = cmocean.cm.thermal
    norm = matplotlib.colors.Normalize(vmin=0, vmax=19, clip = True)
    theory_colors = [color_map(norm(L)) for L in range(k_L_E_arrays.shape[0])]
    theory_formattings = [ {'color': color, 'linewidth': 1.5} for color in theory_colors ]
    theory_distinguished_formattings = [ {'color': 'k', 'linewidth': 4}, ]

    for E_index in range(k_L_E_arrays.shape[1]):
        energy = energy_tuple[E_index]
        png_path = plots_dir_path / 'paper' / f'{journal_name}' / 'SupplementaryFigure1' / f'{input_dir_name}' / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'Fig3_{energy:.2e}K.png'
        pdf_path = png_path.with_suffix('.pdf')
        svg_path = png_path.with_suffix('.svg')
        png_path.parent.mkdir(parents = True, exist_ok = True)
        
        total_k_vs_Phi_at_E_array = np.array([total_k_E_Phis_array[E_index],]).transpose()
        print(f'{total_k_vs_Phi_at_E_array = }')
        theory = k_L_E_arrays[:,E_index,:].transpose()

        fig = plt.figure(figsize=figsize, dpi = dpi)
        ax = fig.add_subplot()
        ax = ValuesVsModelParameters.plotValuestoAxis(ax, xx = singlet_phases, theory = theory, theory_distinguished = total_k_vs_Phi_at_E_array, theory_formattings = theory_formattings, theory_distinguished_formattings = theory_distinguished_formattings)

        ax.set_xlabel(r"$\Phi_\mathrm{s}$", fontsize = 'large')
        ax.set_ylabel('rate ($\\mathrm{cm}^3/\\mathrm{s}$)', fontsize = 'large')
        # ax.set_ylim(0, 1.2*np.amax(total_k_vs_Phi_at_E_array) )
        PhaseTicks.setInMultiplesOfPhi(ax.xaxis)

        # find the maximum for each partial wave and return tuples of the form (L, Phis_max, k_max)
        # for L in range(k_L_E_arrays.shape[0]):
        #     print(f'{L =}')
        #     print(np.any(filter_max_arr[L, E_index]))
        #     print(np.any(k_L_E_arrays[L, E_index][filter_max_arr[L, E_index]] > 0.05*np.nanmax(k_L_E_arrays[:,E_index,:])) )
        #     print(np.any(k_L_E_arrays[L, E_index][filter_max_arr[L, E_index]] > 0.05*np.nanmax(total_k_vs_Phi_at_E_array)) )

        # coords_vs_L = tuple( (l, singlet_phases[filter_max_arr[l, E_index]], k_L_E_arrays[l, E_index][filter_max_arr[l, E_index]]) for l in range(k_L_E_arrays.shape[0]) if np.any(filter_max_arr[l, E_index]) and np.any(k_L_E_arrays[l, E_index][filter_max_arr[l, E_index]] > 0.05*np.nanmax(k_L_E_arrays[:,E_index,:])) )
        # print(f'old: {coords_vs_L = }')
        coords_vs_L = tuple( (l, singlet_phases[filter_max_arr[l, E_index]], k_L_E_arrays[l, E_index][filter_max_arr[l, E_index]]) for l in range(k_L_E_arrays.shape[0]) if np.any(filter_max_arr[l, E_index]) and np.any(k_L_E_arrays[l, E_index][filter_max_arr[l, E_index]] > 0.05*np.nanmax(total_k_vs_Phi_at_E_array)) )
        print(f'{coords_vs_L = }')

        # annotate peaks with the orbital quantum numbers L
        for coord in coords_vs_L:
            ax.text(coord[1], coord[2] + (ax.get_ylim()[1]-ax.get_ylim()[0])*0.02, f'{coord[0]}', fontsize = 'large', color = color_map(norm(coord[0])), fontweight = 'bold', va = 'center', ha = 'center')
        # ax.set_title(f'The $\\left|1,-1\\right>\\hspace{{0.2}}\\left|\\hspace{{-.2}}\\uparrow\\hspace{{-.2}}\\right> \\rightarrow \left|1,0\\right>\\hspace{{0.2}}\\left|\\hspace{{-.2}}\\downarrow\\hspace{{-.2}}\\right>$ collision rate.$')
        # props = dict(boxstyle='round', facecolor='none', edgecolor='none')
        ax.text(0.95, 0.90, f'$\\Delta\\Phi = {phase_difference%1:.2f}\\pi$\n$E_\\mathrm{{col}} = {energy:.2e}\\,\\mathrm{{K}}\\times k_B$', va = 'center', ha = 'right', transform = ax.transAxes)
        # ax.text(0.95, 0.90, f'$E_\\mathrm{{col}} = {energy:.2e}\\,\\mathrm{{K}}\\times k_B$', va = 'center', ha = 'right', transform = ax.transAxes)
        fig.savefig(png_path, bbox_inches='tight', pad_inches = 0)
        fig.savefig(svg_path, bbox_inches='tight', pad_inches = 0, transparent = True)
        fig.savefig(pdf_path, bbox_inches='tight', pad_inches = 0, transparent = True)

        plt.close()

    for zipped_dir_path in zipped_dir_paths:
        zip_path = zipped_dir_path.parent / (zipped_dir_path.name + '.zip')
        base_dir = zipped_dir_path / f'in_2_-2_1_1'
        with ZipFile(zip_path, 'w') as zip:
            for filepath in [*base_dir.rglob('k_*'), *base_dir.rglob('probabilities/*'),]:
                # print(filepath)
                zip.write(filepath)

        # shutil.make_archive(zipped_dir_path, 'zip', root_dir = zipped_dir_path, base_dir = f'in_2_-2_1_1/k_L_E')
        # [shutil.rmtree(zipped_dir_path / f'in_{F1}_{MF1}_{F2}_{MF2}' / name, ignore_errors=True) for name in ('k_L_E', 'k_m_L_E') ]


def main():
    parser_description = "This is a python script for running molscat, collecting and pickling S-matrices, and calculating effective probabilities."
    parser = argparse.ArgumentParser(description=parser_description)
    parser.add_argument("--phase_step", type = float, default = 0.04, help = "The phase step in the multiples of pi for the color map.")
    parser.add_argument("--phase_difference", type = float, default = 0.20, help = "The distinguished value of the singlet-triplet semiclassical phase difference modulo pi in multiples of pi.")
    parser.add_argument("--so_scaling", type = float, default = 0.32, help = "Value of the SO scaling.")

    parser.add_argument("--MF_in", type = int, default = -2)
    parser.add_argument("--MS_in", type = int, default = 1)

    parser.add_argument("--nenergies", type = int, default = 50, help = "Number of energy values in a grid.")
    parser.add_argument("--E_min", type = float, default = 8e-7, help = "Lowest energy value in the grid.")
    parser.add_argument("--E_max", type = float, default = 8e-2, help = "Highest energy value in the grid.")
    parser.add_argument("--n_grid", type = int, default = 3, help = "n parameter for the nth-root energy grid.")


    parser.add_argument("--input_dir_name", type = str, default = 'RbSr+_fmf_so_scaling', help = "Name of the directory with the molscat inputs")
    parser.add_argument("--plot_nan", action = 'store_true', help = "If included, the plotted values will be interpolated for arrays that weren't found (instead of jus plotting a blank place).")

    parser.add_argument("--journal", type = str, default = 'NatCommun', help = "Name of the journal to prepare the plots for.")
    args = parser.parse_args()

    nenergies, E_min, E_max, n = args.nenergies, args.E_min, args.E_max, args.n_grid
    energy_tuple = tuple( round(n_root_scale(i, E_min, E_max, nenergies-1, n = n), sigfigs = 11) for i in range(nenergies) )


    plotRateVsPhisForEachEnergy(phase_step = args.phase_step, phase_difference = args.phase_difference, so_scaling = args.so_scaling, energy_tuple = energy_tuple, input_dir_name = args.input_dir_name, plot_nan = args.plot_nan, journal_name = args.journal)


if __name__ == '__main__':
    main()