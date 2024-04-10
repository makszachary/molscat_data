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


def latex_scientific_notation(number, sigfigs = 2):
    float_str = f'{number:.{sigfigs}e}'
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return f'{base} \\times 10^{{{int(exponent)}}}'
    else:
        return float_str

def plotRateVsPhisForEachEnergy(phase_step: float, phase_difference: float, so_scaling: float, energy_tuple: tuple[float, ...], plot_energies: np.ndarray = None, input_dir_name: str = 'RbSr+_fmf_so_scaling', abbreviation = 'cold_lower', L_max: int = 2*49, plot_nan = False, merge_plots = False, journal_name = 'NatCommun'):
    plt.style.use(Path(__file__).parent / 'mpl_style_sheets' / f'{journal_name}.mplstyle')

    time_0 = time.perf_counter()
    cm = 1/2.54
    total_height = len(plot_energies)*4.7 if plot_energies is not None else 6
    figsize = (15*cm, total_height*cm)
    dpi = 1200

    molscat_out_dir = scratch_path.joinpath('molscat', 'outputs')

    nenergies = len(energy_tuple)
    E_min = min(energy_tuple)
    E_max = max(energy_tuple)
    # singlet_phases, triplet_phases = np.array(singlet_phases), np.array(triplet_phases)

    if abbreviation == 'cold_lower':
        F1in, MF1in, F2in, MF2in = 2, -2, 1, 1
        F1out, MF1out, F2out, MF2out = 2, 0, 1, -1
    elif abbreviation == 'cold_higher':
        F1in, MF1in, F2in, MF2in = 4, -4, 1, 1
        F1out, MF1out, F2out, MF2out = 4, -2, 1, -1
    elif abbreviation == 'hpf':
        F1in, MF1in, F2in, MF2in = 4, -4, 1, 1
        F1out, MF1out, F2out, MF2out = 2, -2, 1, -1
    else:
        raise ValueError(f"{abbreviation = } doesn\'t match any of the following: \'cold_lower\', \'cold_higher\', \'hpf\'.")

    ## SECTIONS THROUGH THE CONTOUR MAP

    singlet_phases = np.array([default_singlet_phase_function(1.0),]) if phase_step is None else np.arange(phase_step, 1., phase_step).round(decimals=4)

    zipped_dir_paths = [ arrays_dir_path / input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{(singlet_phase+phase_difference)%1:.4f}' / f'{so_scaling:.4f}' for singlet_phase in singlet_phases if ( singlet_phase+phase_difference ) % 1 !=0]
    for zipped_dir_path in zipped_dir_paths:
        zip_path = zipped_dir_path.parent / (zipped_dir_path.name + '.zip')
        if zip_path.is_file():        
            shutil.unpack_archive(zip_path, zipped_dir_path, 'zip')
        else:
            print(f'{zip_path = } is not a file.')

    k_L_E_array_paths = [  arrays_dir_path / input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{(singlet_phase+phase_difference)%1:.4f}' / f'{so_scaling:.4f}' / f'in_{F1in}_{MF1in}_{F2in}_{MF2in}' / 'k_L_E' / abbreviation / f'OUT_{F1out}_{MF1out}_{F2out}_{MF2out}_IN_{F1in}_{MF1in}_{F2in}_{MF2in}.txt' if ( singlet_phase+phase_difference ) % 1 !=0 else None for singlet_phase in singlet_phases]
    [print(f'{array_path} not found') for array_path in k_L_E_array_paths if (array_path is not None and not array_path.is_file())]
    print(k_L_E_array_paths)
    k_L_E_arrays = [np.loadtxt(array_path) if (array_path is not None and array_path.is_file()) else np.full((int(L_max/2+1), nenergies), np.nan) for array_path in k_L_E_array_paths]
    k_L_E_shapes = [arr.shape for arr in k_L_E_shapes]
    print(f'{k_L_E_shapes}')
    k_L_E_arrays = np.array(k_L_E_arrays).transpose(1,2,0)
    if plot_nan:
        print(f'{k_L_E_arrays[np.isnan(k_L_E_arrays)] = }')
        print(f'{k_L_E_arrays[np.roll(np.isnan(k_L_E_arrays),1,2)] = }')
        print(f'{k_L_E_arrays[np.roll(np.isnan(k_L_E_arrays),-1,2)] = }')
        k_L_E_arrays[np.isnan(k_L_E_arrays)] = (k_L_E_arrays[np.roll(np.isnan(k_L_E_arrays),-1,2)]+k_L_E_arrays[np.roll(np.isnan(k_L_E_arrays),1,2)])/2
        print(f'{k_L_E_arrays[np.isnan(k_L_E_arrays)] = }')
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
    theory_formattings = [ {'color': color, 'linewidth': 1.25} for color in theory_colors ]
    theory_distinguished_formattings = [ {'color': 'k', 'linewidth': 2.0}, ]
    
    energy_indices = tuple( (np.abs(energy_tuple - value)).argmin() for value in plot_energies) if plot_energies is not None else tuple(range(k_L_E_arrays.shape[1]))

    if merge_plots and plot_energies is not None:
        png_path = plots_dir_path / 'paper' / f'{journal_name}' / 'SupplementaryFigure1_merged' / f'{input_dir_name}' / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / abbreviation / f'SupplementaryFig1_{abbreviation}.png'
        pdf_path = png_path.with_suffix('.pdf')
        svg_path = png_path.with_suffix('.svg')
        png_path.parent.mkdir(parents = True, exist_ok = True)
        
        fig, axs = plt.subplots(3, 1, sharex = True, figsize = figsize, dpi = dpi)
        fig.subplots_adjust(hspace=0.05)
        # fig = plt.figure(figsize=figsize, dpi = dpi)
        # gs = gridspec.GridSpec(3,1, fig)
        # gs.update(hspace=0.0)
        # axs = [fig.add_subplot(gs[i]) for i in range(len(plot_energies))]
        for E_index, ax in zip(energy_indices, axs):
            energy = energy_tuple[E_index]
            total_k_vs_Phi_at_E_array = np.array([total_k_E_Phis_array[E_index],]).transpose()
            theory = k_L_E_arrays[:,E_index,:].transpose()

            ax = ValuesVsModelParameters.plotValuestoAxis(ax, xx = singlet_phases, theory = theory, theory_distinguished = total_k_vs_Phi_at_E_array, theory_formattings = theory_formattings, theory_distinguished_formattings = theory_distinguished_formattings)
            ax.set_ylim(0, 1.2*np.amax(total_k_vs_Phi_at_E_array))
            PhaseTicks.setInMultiplesOfPhi(ax.xaxis)
            preferred_exponent = -9
            ax.yaxis.set_major_formatter(lambda x, pos: f'{x / 10**(preferred_exponent):.1f}')
            ax.set_ylabel(f'$k_\\mathrm{{SE}}$ ($\\,10^{{{int(preferred_exponent)}}}\\,\\mathrm{{cm}}^3/\\mathrm{{s}}$)')

            # find the maximum for each partial wave and return tuples of the form (L, Phis_max, k_max)
            coords_vs_L = tuple( (l, singlet_phases[filter_max_arr[l, E_index]], k_L_E_arrays[l, E_index][filter_max_arr[l, E_index]]) for l in range(k_L_E_arrays.shape[0]) if np.any(filter_max_arr[l, E_index]) and np.any(k_L_E_arrays[l, E_index][filter_max_arr[l, E_index]] > 0.05*np.nanmax(total_k_vs_Phi_at_E_array)) )

            # annotate peaks with the orbital quantum numbers L
            for coord in coords_vs_L:
                ax.text(coord[1], coord[2] + (ax.get_ylim()[1]-ax.get_ylim()[0])*0.02, f'{coord[0]}', fontsize = 'large', color = color_map(norm(coord[0])), fontweight = 'bold', va = 'center', ha = 'center')
            ax.text(0.97, 0.915, f'$E_\\mathrm{{col}} = {latex_scientific_notation(energy)}\\,\\mathrm{{K}}\\times k_B$', va = 'top', ha = 'right', transform = ax.transAxes)#, bbox = dict(facecolor = 'white', edgecolor = 'none'))

        for i in range(len(axs)-1):
            plt.setp(axs[i].get_xticklabels(), visible=False)
        axs[-1].set_xlabel(r"$\Phi_\mathrm{s}$")

        for i, ax in enumerate(axs):
            ax.text(-0.08, 1.0, f'{chr(ord("a")+i)}', fontsize = 8, family = 'sans-serif', va = 'top', ha = 'left', transform = ax.transAxes, fontweight = 'bold')
        fig.savefig(png_path, bbox_inches='tight', pad_inches = 0)
        fig.savefig(svg_path, bbox_inches='tight', pad_inches = 0, transparent = True)
        fig.savefig(pdf_path, bbox_inches='tight', pad_inches = 0, transparent = True)
        print(f"Merged plots created as {pdf_path} etc.")
        return

    for E_index in energy_indices:#range(k_L_E_arrays.shape[1]):
        energy = energy_tuple[E_index]
        png_path = plots_dir_path / 'paper' / f'{journal_name}' / 'SupplementaryFigure1' / f'{input_dir_name}' / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / abbreviation / f'SupplementaryFig1_{abbreviation}_{energy:.2e}K.png'
        pdf_path = png_path.with_suffix('.pdf')
        svg_path = png_path.with_suffix('.svg')
        png_path.parent.mkdir(parents = True, exist_ok = True)
        
        total_k_vs_Phi_at_E_array = np.array([total_k_E_Phis_array[E_index],]).transpose()
        print(f'{total_k_vs_Phi_at_E_array = }')
        theory = k_L_E_arrays[:,E_index,:].transpose()

        fig = plt.figure(figsize=figsize, dpi = dpi)
        ax = fig.add_subplot()
        ax = ValuesVsModelParameters.plotValuestoAxis(ax, xx = singlet_phases, theory = theory, theory_distinguished = total_k_vs_Phi_at_E_array, theory_formattings = theory_formattings, theory_distinguished_formattings = theory_distinguished_formattings)

        # ax.set_ylim(0, 1.2*np.amax(total_k_vs_Phi_at_E_array) )
        PhaseTicks.setInMultiplesOfPhi(ax.xaxis)
        ax.set_xlabel(r"$\Phi_\mathrm{s}$")
        preferred_exponent = -9
        # ax.ticklabel_format(axis = 'y', scilimits = (-9,-9))
        ax.yaxis.set_major_formatter(lambda x, pos: f'{x / 10**(preferred_exponent):.1f}')
        # base, exponent = f'{ax.get_ylim()[1]:.2e}'.split("e")
        ax.set_ylabel(f'rate ($\\times\\,10^{{{int(preferred_exponent)}}}\\,\\mathrm{{cm}}^3/\\mathrm{{s}}$)')

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
        # ax.text(0.95, 0.90, f'$\\Delta\\Phi = {phase_difference%1:.2f}\\pi$\n$E_\\mathrm{{col}} = {energy:.2e}\\,\\mathrm{{K}}\\times k_B$', va = 'center', ha = 'right', transform = ax.transAxes)
        ax.text(0.95, 0.90, f'$E_\\mathrm{{col}} = {latex_scientific_notation(energy)}\\,\\mathrm{{K}}\\times k_B$', va = 'center', ha = 'right', transform = ax.transAxes)
        # ax.text(0.95, 0.90, f'$E_\\mathrm{{col}} = {energy:.2e}\\,\\mathrm{{K}}\\times k_B$', va = 'center', ha = 'right', transform = ax.transAxes)
        fig.savefig(png_path, bbox_inches='tight', pad_inches = 0)
        fig.savefig(svg_path, bbox_inches='tight', pad_inches = 0, transparent = True)
        fig.savefig(pdf_path, bbox_inches='tight', pad_inches = 0, transparent = True)

        plt.close()

    for zipped_dir_path in zipped_dir_paths:
        zip_path = zipped_dir_path.parent / (zipped_dir_path.name + '.zip')
        base_dir = zipped_dir_path / f'in_2_-2_1_1'
        with ZipFile(zip_path, 'w') as zipfl:
            for filepath in [*base_dir.rglob('k_*'), *base_dir.rglob('probabilities/*'),]:
                # print(filepath)
                zipfl.write(filepath, arcname = filepath.relative_to(zipped_dir_path))

        # shutil.make_archive(zipped_dir_path, 'zip', root_dir = zipped_dir_path, base_dir = f'in_2_-2_1_1/k_L_E')
        # [shutil.rmtree(zipped_dir_path / f'in_{F1}_{MF1}_{F2}_{MF2}' / name, ignore_errors=True) for name in ('k_L_E', 'k_m_L_E') ]


def main():
    parser_description = "This is a python script for running molscat, collecting and pickling S-matrices, and calculating effective probabilities."
    parser = argparse.ArgumentParser(description=parser_description)
    parser.add_argument("--phase_step", type = float, default = 0.04, help = "The phase step in the multiples of pi for the color map.")
    parser.add_argument("--phase_difference", type = float, default = 0.20, help = "The distinguished value of the singlet-triplet semiclassical phase difference modulo pi in multiples of pi.")
    parser.add_argument("--so_scaling", type = float, default = 0.32, help = "Value of the SO scaling.")

    # parser.add_argument("--MF_in", type = int, default = -2)
    # parser.add_argument("--MS_in", type = int, default = 1)
    parser.add_argument("--abbreviation", type = str, default = 'cold_lower', help = "The name of abbreviation in the array path. Possible values: \'cold_lower\' (default), \'cold_higher\', \'hpf\'.")
    parser.add_argument("--L_max", type = int, default = 49, help = "DOUBLED maximum orbital angular momentum quantum number. Default: 2*49.")

    parser.add_argument("--nenergies", type = int, default = 50, help = "Number of energy values in a grid.")
    parser.add_argument("--E_min", type = float, default = 8e-7, help = "Lowest energy value in the grid.")
    parser.add_argument("--E_max", type = float, default = 8e-2, help = "Highest energy value in the grid.")
    parser.add_argument("--n_grid", type = int, default = 3, help = "n parameter for the nth-root energy grid.")

    parser.add_argument("--plot_energies", nargs='*', type = float, default = None, help = "Values of the energies to plot.")
    parser.add_argument("--merge_plots", action = 'store_true', help = "If included, common figure with subplots will be plotted, not separate figures.")

    parser.add_argument("--input_dir_name", type = str, default = 'RbSr+_fmf_so_scaling', help = "Name of the directory with the molscat inputs")
    parser.add_argument("--plot_nan", action = 'store_true', help = "If included, the plotted values will be interpolated for arrays that weren't found (instead of jus plotting a blank place).")

    parser.add_argument("--journal", type = str, default = 'NatCommun', help = "Name of the journal to prepare the plots for.")
    args = parser.parse_args()

    nenergies, E_min, E_max, n = args.nenergies, args.E_min, args.E_max, args.n_grid
    energy_tuple = tuple( round(n_root_scale(i, E_min, E_max, nenergies-1, n = n), sigfigs = 11) for i in range(nenergies) )

    plot_energies = None if args.plot_energies is None else np.array(args.plot_energies)

    plotRateVsPhisForEachEnergy(phase_step = args.phase_step, phase_difference = args.phase_difference, so_scaling = args.so_scaling, energy_tuple = energy_tuple, plot_energies = plot_energies, input_dir_name = args.input_dir_name, abbreviation = args.abbreviation, L_max = args.L_max, plot_nan = args.plot_nan, merge_plots = args.merge_plots, journal_name = args.journal)


if __name__ == '__main__':
    main()