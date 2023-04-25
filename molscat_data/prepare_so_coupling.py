import numpy as np
from pathlib import Path

import argparse

from _molscat_data.physical_constants import hartree_in_inv_cm, fine_structure_constant

def scale_so_and_write(input_path: Path | str, output_path: Path | str, scaling: float, header: str = None) -> None:
    """Scales the spin-orbit term provided in input_path (in cm-1),
    adds the spin-spin term, and writes to output_path in a format
    needed in RKHS routine. The units in POTENL in molscat input
    should be set to atomic units (Hartees and bohrs).

    :param input_path: Path to a file providing the second-order spin-orbit
    coupling at short range in cm-1 and bohrs.
    :param output_path: Path where to write the output.
    :param scaling: Scaling of the short-range part of the provided data.
    """
    
    Path(output_path).parent.mkdir(parents = True, exist_ok = True)

    so_coupling = np.loadtxt(input_path, dtype = float)
    
    C3 = 1/hartree_in_inv_cm
    number_of_points = so_coupling.shape[0]
    max_distance = np.max(so_coupling[:,0])
    condition_check_distance = max_distance + 50
    
    if header == None:
        header = (  f'# first line anyway ignored (n. of points, energy shift, scaling = scaling x scaling, unit of disctance, unit of energy [both in units defined in POTL block])\n'
                    f'{number_of_points} 0.0 1.0 1.0 1.0\n'
                    f'# also ignored (3 integer parameters n, m, and s controling the RKHS, if we want to impose conditions, number of coefficients we want to impose conditions on)\n'
                    f'3 2 1 T 1\n'
                    f'# also ignored (Ra, C3, RC3)\n'
                    f'25.0 {C3:.15e} {condition_check_distance}\n'
                    f'# also ignored'
                    )
    

    so_coupling[:,1] /= (hartree_in_inv_cm**2 * fine_structure_constant**2)
    so_coupling[:,1] *= scaling
    so_coupling[:,1] -=  C3/so_coupling[:,0]**3
    np.savetxt(output_path, so_coupling, fmt = ['%.2f','%.15e'], header = header, comments = '')


def main():
    parser_description = f"""This script prepares the input for RKHS routine in molscat
    for the given second-order spin-orbit coupling in the short range
    and the scaling of its short-range part. The pure spin-spin interaction
    term is added."""
    parser = argparse.ArgumentParser(description=parser_description)
    parser.add_argument("-i", "--input", type = str, required = True, help = "Path to the input file or directory")
    parser.add_argument("-o", "--output", type = str, required = True, help = "Path to the output file or directory")
    parser.add_argument("-s", "--scaling", type = float, default = 1., help = "Scaling of the short-range part of the potential.")
    args = parser.parse_args()

    scale_so_and_write(args.input, args.output, args.scaling)

if __name__ == '__main__':
    main()