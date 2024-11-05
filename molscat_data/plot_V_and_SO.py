import numpy as np
from _molscat_data.scaling_old import read_from_json, default_singlet_phase_function, default_triplet_phase_function, default_singlet_parameter_from_phase, default_triplet_parameter_from_phase
from matplotlib import pyplot as plt
import matplotlib
from scipy.interpolate import interp1d
import os
from pathlib import Path
import argparse
import time
from math import log10, floor
from labellines import labelLines, labelLine

from _molscat_data.physical_constants import red_mass_87Rb_88Sr_amu, hartree_in_inv_cm, fine_structure_constant
from _molscat_data.visualize import PhaseTicks

# filepath = r"C:\Users\maksw\Documents\python\data\SO\RKHS\molscat-RbSr+30.json"
# impath = filepath.strip('.json')+'.png'
# singletpotential, tripletpotential = read_from_json(filepath)

@matplotlib.ticker.FuncFormatter
def exponential_major_formatter(x, pos, ndp = 0):
    if float(x) == 0:
        return '0'
    s = '{x:0.{ndp:d}e}'.format(x=x, ndp=ndp)
    m, e = s.split('e')
    return r'${m:s}\times 10^{{{e:d}}}$'.format(m=m, e=int(e))


def plot_potentials(file_path: Path | str, impath: Path | str = None, show: bool = False, original_path: Path | str = None, journal_name = 'NatCommun',) -> None:
    plt.style.use(Path(__file__).parent / 'mpl_style_sheets' / f'{journal_name}.mplstyle')
    singletpotential, tripletpotential, so_coupling = read_from_json(file_path)

    ### getting lambda_so (in Hartrees)
    so_coupling['energy'] = np.array(so_coupling['energy']) + fine_structure_constant**2 / np.array(so_coupling['distance'])**3
    print(np.amin(so_coupling['energy']))
    
    # singlet_De = np.amin(singletpotential['energy'])
    # singlet_Re = np.array(singletpotential['distance'])[np.array(singletpotential['energy']) == singlet_De]
    # triplet_De = np.amin(tripletpotential['energy'])
    # triplet_Re = np.array(tripletpotential['distance'])[np.array(tripletpotential['energy']) == triplet_De]
    # print(f'{singlet_De = }\n{np.mean(singlet_Re) = }\n{triplet_De = }\n{np.mean(triplet_Re) = }')

    cm = 1/2.54
    nrows = 1
    row_height = 6
    vpad = 1
    total_height = nrows*row_height + (nrows-1)*vpad
    # figsize = (18*cm, total_height*cm)
    figsize = (8.8*cm, total_height*cm)
    dpi = 1000

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax_so_coupling = ax.twinx()

    ax.plot(singletpotential['distance'], np.array(singletpotential['energy'])*hartree_in_inv_cm, color = 'tab:blue', linewidth = 1, linestyle = '--', label = "$(2)\,{}^{1}\Sigma^{+}$")
    ax.plot(tripletpotential['distance'], np.array(tripletpotential['energy'])*hartree_in_inv_cm, color = 'tab:purple', linewidth = 1, linestyle = '--', label = "$(1)\,{}^{3}\Sigma^{+}$")
    ax_so_coupling.plot(so_coupling['distance'], np.array(so_coupling['energy'])*hartree_in_inv_cm, color = 'black', linewidth = 1, linestyle = '--', label = "$\lambda_\mathrm{so}$ (fitted)")
    if original_path is not None and not Path(original_path).is_file():
        raise ValueError("The given original_path is not a file.")
    if Path(original_path).is_file():
            original_singletpotential, original_tripletpotential, original_so_coupling = read_from_json(original_path)
            ### getting lambda_so (in Hartrees)
            original_so_coupling['energy'] = np.array(original_so_coupling['energy']) + fine_structure_constant**2 / np.array(original_so_coupling['distance'])**3
            print(np.amin(original_so_coupling['energy']))
            ax.plot(original_singletpotential['distance'], np.array(original_singletpotential['energy'])*hartree_in_inv_cm, color = 'tab:blue', linewidth = 1, )
            ax.plot(original_tripletpotential['distance'], np.array(original_tripletpotential['energy'])*hartree_in_inv_cm, color = 'tab:purple', linewidth = 1, )
            ax_so_coupling.plot(original_so_coupling['distance'], np.array(original_so_coupling['energy'])*hartree_in_inv_cm, color = 'black', linewidth = 1, label = "$\lambda_\mathrm{so}$ (ab initio)")
    # plt.plot(singletpotential['distance'], np.array(singletpotential['distance'])**4 * np.array(singletpotential['energy']), color = 'tab:blue', label = "$A^{1}\Sigma^{+}$")
    # plt.plot(tripletpotential['distance'], np.array(tripletpotential['distance'])**4 * np.array(tripletpotential['energy']), color = 'tab:purple', label = "$a^{3}\Sigma^{+}$")
    ax.set_xlim(5, 25)
    # ax.set_ylim(-3e-2*hartree_in_inv_cm, 5e-2*hartree_in_inv_cm)
    ax.set_ylim(-8e3, 10e3)
    ax_so_coupling.set_ylim(np.array(ax.get_ylim())/5e3)
    ax.set_xlabel("$R$ ($a_0$)")
    ax.set_ylabel("$V$ ($\mathrm{cm}^{-1}$)")
    ax_so_coupling.set_ylabel("$\lambda_{so}$ ($\mathrm{cm}^{-1}$)")
    # plt.ylabel("$V(R) \cdot R^4$, ($E_h$)", fontsize = 'xx-large')
    # fig.legend()

    PhaseTicks.linearStr(ax.xaxis, 5, 1, '${x:.0f}$')
    PhaseTicks.linearStr(ax.yaxis, 5e3, 1e3, '${x:.0f}$')
    PhaseTicks.linearStr(ax_so_coupling.yaxis, 1, 0.2, '${x:.0f}$')
    # ax_so_coupling.yaxis.set_major_formatter(exponential_major_formatter)
    ax.tick_params(axis = 'both', which = 'both', direction = 'in')
    ax_so_coupling.tick_params(axis = 'both', which = 'both', direction = 'in')

    # labelLines(ax.get_lines(), align = False, outline_width=2, color = 'white', fontsize = matplotlib.rcParams["xtick.labelsize"], zorder = 3)
    labelLines(ax.get_lines(), align = False, outline_color=None, fontsize = matplotlib.rcParams["xtick.labelsize"], zorder = 3)
    labelLines(ax_so_coupling.get_lines(), align = False, outline_color=None, fontsize = matplotlib.rcParams["xtick.labelsize"], zorder = 3)

    plt.tight_layout()
    if impath is not None:
        plt.savefig(impath)
    if show == True:
        plt.show()
    else:
        plt.close()

def plot_so_potdiff_centrifugal(file_path: Path | str, impath: Path | str = None, L: int = 2, reduced_mass: float = red_mass_87Rb_88Sr_amu, show: bool = False) -> None:
    singlet_potential, triplet_potential, so_coupling = read_from_json(file_path)
    plt.figure()
    plt.plot(singlet_potential['distance'], np.abs(np.array(triplet_potential['energy'])-np.array(singlet_potential['energy'])), color = 'tab:blue', label = "$|V_t(R)-V_s(R)|$")
    plt.plot(so_coupling['distance'], so_coupling['energy'], color = 'black', label = "$\lambda_\mathrm{SO+SS}(R)$")
    plt.plot(np.array(so_coupling['distance']), L*(L+1)/(2 * reduced_mass * np.array(so_coupling['distance']) ), color = 'red', label = f"$L(L+1)/2 \mu R^2$ for {L=}" )
    # max_so = max(so_coupling['energy'])
    max_so = 1e-5
    plt.xlim(5, 60)
    plt.ylim(-0.1*max_so, 2*max_so)
    plt.xlabel("$R, a_0$", fontsize = 'xx-large')
    plt.ylabel("$V(R)$, ($E_h$)", fontsize = 'xx-large')
    plt.grid('both')
    plt.legend()
    plt.tight_layout()
    if impath is not None:
        Path(impath).parent.mkdir(parents = True, exist_ok = True)
        plt.savefig(impath)
    if show == True:
        plt.show()
    else:
        plt.close()

def plot_scaling(path, xrange = [11,20], yrange = [-0.004, 0.0001], figsize=(10,6), dpi = 100, impath = None, show = False):
    xx = np.arange(6,50,0.01)
    data = read_from_json(path)
    singletdata, tripletdata = [], []
    for item in data:
        if item['label'] == 'singlet':
            singletdata.append(item)
        elif item['label'] == 'triplet':
            tripletdata.append(item)
    singletdata = sorted(singletdata, key = lambda i: ( i['scaling'] ) )
    fs0 = interp1d(singletdata[0]['distance'], singletdata[0]['energy'])
    fs1 = interp1d(singletdata[-1]['distance'], singletdata[-1]['energy'])
    tripletdata = sorted(tripletdata, key = lambda i: ( i['scaling'] ) )
    ft0 = interp1d(tripletdata[0]['distance'], tripletdata[0]['energy'])
    ft1 = interp1d(tripletdata[-1]['distance'], tripletdata[-1]['energy'])
    
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(100, 100, (1,10000))

    linecolor = 'firebrick'
    fillcolor = 'darksalmon'
    ax.plot(xx, fs0(xx), color = linecolor, linewidth = 2, linestyle = '--', label = r"$(2)\,{}^{1}\Sigma^{+}$")
    ax.plot(xx, fs1(xx), color = linecolor, linewidth = 2, linestyle = '--')
    ax.fill_between(xx, fs0(xx), fs1(xx), color = fillcolor)

    ax1 = fig.add_subplot(100, 100, (2865,6100))
    ax1.plot(xx, fs0(xx), color = linecolor, linestyle = '--')
    ax1.plot(xx, fs1(xx), color = linecolor, linestyle = '--')
    ax1.fill_between(xx, fs0(xx), fs1(xx), color = fillcolor)
    ax1.set_xlim(12.5, 16.8)
    ax1.set_ylim(-0.0038,-0.0022)
    ax1.tick_params(axis = 'both', direction='in', pad = 2, grid_color = 'gray', grid_alpha = 0.5)
    ax1.tick_params(axis = 'x', top = True, labeltop = True, labelbottom = False)
    ax1.grid()

    linecolor = 'indigo'
    fillcolor = 'mediumslateblue'
    ax.plot(xx, ft0(xx), color = linecolor, linewidth = 2, linestyle = '--', label = r"$(1)\,{}^{3}\Sigma^{+}$")
    ax.plot(xx, ft1(xx), color = linecolor, linewidth = 2, linestyle = '--')
    ax.fill_between(xx, ft0(xx), ft1(xx), color = fillcolor)

    ax2 = fig.add_subplot(100, 100, (6265,9500))
    ax2.plot(xx, ft0(xx), color = linecolor, linestyle = '--')
    ax2.plot(xx, ft1(xx), color = linecolor, linestyle = '--')
    ax2.fill_between(xx, ft0(xx), ft1(xx), color = fillcolor)
    ax2.set_xlim(8.1, 12.4)
    ax2.set_ylim(-0.0295,-0.0225)
    ax2.tick_params(axis = 'both', direction='in', pad = 2, grid_color = 'gray', grid_alpha = 0.5)
    ax2.tick_params(axis = 'x', top = True)
    ax2.grid()

    # for tripletpotential in tripletdata:
    #     ax.plot(tripletpotential['distance'], tripletpotential['energy'], color = 'tab:purple', label = r"$a^{3}\Sigma^{+} \times %s$" % tripletpotential['scaling'])
    # ax.plot(singletpotential['distance'], np.array(singletpotential['distance'])**4 * np.array(singletpotential['energy']), color = 'tab:blue', label = "$A^{1}\Sigma^{+}$")
    # ax.plot(tripletpotential['distance'], np.array(tripletpotential['distance'])**4 * np.array(tripletpotential['energy']), color = 'tab:purple', label = "$a^{3}\Sigma^{+}$")
    ax.set_xlim(xrange)
    ax.set_ylim(yrange)
    ax.tick_params(axis = 'both', labelsize ='large')
    ax.set_xlabel("$R \, (a_0)$", fontsize = 'xx-large')
    ax.set_ylabel("$V(R) \, (E_h$)", fontsize = 'xx-large')
    # plt.ylabel("$V(R) \cdot R^4$, ($E_h$)", fontsize = 'xx-large')
    ax.grid(color = 'gray')
    fig.legend(loc = 'upper right', bbox_to_anchor = (0.90, 0.89), labelspacing = 0.1, fontsize = 'large')
    plt.tight_layout()
    if isinstance(impath, str):
        plt.savefig(impath)
    if show == True:
        plt.show()
    else:
        plt.close()

def main():
    parser_description = "This is a python script for plotting the potential curves fitted by RKHS from MOLSCAT outputs."
    parser = argparse.ArgumentParser(description=parser_description)
    parser.add_argument("-i", "--input", type = str, required = True, help = "Path to the input file or directory")
    parser.add_argument("-o", "--output", type = str, required = True, help = "Path to the output file or directory")
    # parser.add_argument("-r", "--recursive", action = 'store_true', help = "If enabled, the input directory will be searched for .output files recursively.")
    parser.add_argument("--extension", type = str, default = r'.png', help = "Extension of the output plot file (if a directory was specified as an output and input)")
    parser.add_argument("--show", action = 'store_true', help = "If enabled, the image will be shown (in the single-file case).")
    parser.add_argument("--scaling", action = 'store_true', help = "If enabled, the image will show all the potential curves to show scaling.")
    parser.add_argument("--original", type = str, default = None, help = "Path to the original input file or directory")

    parser.add_argument("--journal", type = str, default = 'NatCommun', help = "Name of the journal to prepare the plots for.")
    args = parser.parse_args()

    singlet_phases = np.arange(0.01, 1.00, 0.01)
    triplet_phases = np.arange(0.01, 1.00, 0.01)
    Phis_vs_scaling = np.array([singlet_phases, [default_singlet_parameter_from_phase(phase) for phase in singlet_phases]]).transpose()
    Phit_vs_scaling = np.array([triplet_phases, [default_triplet_parameter_from_phase(phase) for phase in triplet_phases]]).transpose()
    singlet_txt_path = Path(__file__).parents[1] / 'data_produced' / 'Phi_vs_scaling' / 'Phis_vs_scaling.txt'
    triplet_txt_path = singlet_txt_path.with_stem('Phit_vs_scaling')
    singlet_txt_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(singlet_txt_path, Phis_vs_scaling, fmt = '%.4f, %.8f')
    np.savetxt(triplet_txt_path, Phit_vs_scaling, fmt = '%.4f, %.8f')
    print(f'{default_singlet_phase_function(1.00) = }')
    print(f'{default_triplet_phase_function(1.00) = }')


    if args.scaling:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        plot_scaling(path = args.input, xrange = [6,30], yrange = [-0.03, 0.005], impath = args.output, show = args.show)
    elif Path(args.input).is_file() and not Path(args.output).is_dir():
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        plot_potentials(file_path = args.input, impath = args.output, show = args.show, original_path = args.original, journal_name = args.journal,)
    elif Path(args.input).is_dir():
        Path(args.output).mkdir(parents=True, exist_ok=True)
        for file_path in Path(args.input).iterdir():
            if file_path.is_file() and file_path.name.endswith('.json'):
                print(file_path.name)
                plot_potentials(file_path = file_path, impath = Path(args.output).joinpath(file_path.with_suffix(args.extension).name), original_path = args.original, journal_name = args.journal,)
                print(f"Data from {file_path.name} plotted to {Path(args.output).joinpath(file_path.with_suffix(args.extension).name)}.")
    else:
        print("Input and output should both be .json files or both should be directories. Try again")
        return

if __name__ == "__main__":
    start_time = time.time()
    main()
    print("---Time of generating plots was %s seconds ---" % (time.time() - start_time))