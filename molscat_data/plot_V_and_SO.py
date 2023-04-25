import numpy as np
from _molscat_data.scaling_old import read_from_json
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
import os
from pathlib import Path
import argparse
import time
from _molscat_data.physical_constants import red_mass_87Rb_88Sr

# filepath = r"C:\Users\maksw\Documents\python\data\SO\RKHS\molscat-RbSr+30.json"
# impath = filepath.strip('.json')+'.png'
# singletpotential, tripletpotential = read_from_json(filepath)

def plot_potentials(file_path: Path | str, impath: Path | str = None, show: bool = False) -> None:
    singletpotential, tripletpotential, so_coupling = read_from_json(file_path)
    plt.figure()
    plt.plot(singletpotential['distance'], singletpotential['energy'], color = 'tab:blue', label = "$(2)\,{}^{1}\Sigma^{+}$")
    plt.plot(tripletpotential['distance'], tripletpotential['energy'], color = 'tab:purple', label = "$(1)\,{}^{3}\Sigma^{+}$")
    plt.plot(so_coupling['distance'], np.array(so_coupling['energy'])*10**3, color = 'black', label = "$\lambda_\mathrm{SO+SS}(R)\\times10^3$")
    # plt.plot(singletpotential['distance'], np.array(singletpotential['distance'])**4 * np.array(singletpotential['energy']), color = 'tab:blue', label = "$A^{1}\Sigma^{+}$")
    # plt.plot(tripletpotential['distance'], np.array(tripletpotential['distance'])**4 * np.array(tripletpotential['energy']), color = 'tab:purple', label = "$a^{3}\Sigma^{+}$")
    plt.xlim(5, 25)
    plt.ylim(-0.03, 0.05)
    plt.xlabel("$R, a_0$", fontsize = 'xx-large')
    plt.ylabel("$V(R)$, ($E_h$)", fontsize = 'xx-large')
    # plt.ylabel("$V(R) \cdot R^4$, ($E_h$)", fontsize = 'xx-large')
    plt.grid('both')
    plt.legend()
    plt.tight_layout()
    if impath is not None:
        plt.savefig(impath)
    if show == True:
        plt.show()
    else:
        plt.close()

def plot_so_potdiff_centrifugal(file_path: Path | str, impath: Path | str = None, L: int = 2, reduced_mass: float = red_mass_87Rb_88Sr, show: bool = False) -> None:
    singlet_potential, triplet_potential, so_coupling = read_from_json(file_path)
    plt.figure()
    plt.plot(singlet_potential['distance'], np.array(triplet_potential['energy'])-np.array(singlet_potential['energy']), color = 'tab:blue', label = "$(1)\,{}^{3}\Sigma^{+} - (2)\,{}^{1}\Sigma^{+}$")
    plt.plot(so_coupling['distance'], so_coupling['energy'], color = 'black', label = "$\lambda_\mathrm{SO+SS}(R)$")
    plt.plot(np.array(so_coupling['distance']), L*(L+1)/(2 * reduced_mass * np.array(so_coupling['distance']) ) )
    max_so = max(so_coupling['energy'])
    plt.xlim(5, 100)
    plt.ylim(-1.5*max_so, 1.6*max_so)
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
    args = parser.parse_args()


    if args.scaling:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        plot_scaling(path = args.input, xrange = [6,30], yrange = [-0.03, 0.005], impath = args.output, show = args.show)
    elif Path(args.input).is_file() and not Path(args.output).is_dir:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        plot_potentials(file_path = args.input, impath = args.output, show = args.show)
    elif Path(args.input).is_dir():
        Path(args.output).mkdir(parents=True, exist_ok=True)
        for file_path in Path(args.input).iterdir():
            if file_path.is_file() and file_path.name.endswith('.json'):
                print(file_path.name)
                plot_potentials(file_path = file_path, impath = Path(args.output).joinpath(file_path.with_suffix(args.extension).name) )
                print(f"Data from {file_path.name} plotted to {Path(args.output).joinpath(file_path.with_suffix(args.extension).name)}.")
    else:
        print("Input and output should both be .json files or both should be directories. Try again")
        return

if __name__ == "__main__":
    start_time = time.time()
    main()
    print("---Time of generating plots was %s seconds ---" % (time.time() - start_time))