import warnings

import numpy as np

from _molscat_data.scaling_old import update_json
from _molscat_data.physical_constants import hartree_in_inv_cm, fine_structure_constant
from sigfig import round
import argparse
import os
from pathlib import Path
import time

# filepath = r"C:\Users\maksw\Documents\python\data\SO\RKHS\molscat-RbSr+30.output"


def get_potentials_and_so(file_path: Path | str) -> tuple[dict, dict, dict]:

    singletpotential = {
        'label': 'singlet',
        'scaling': None,
        'distance': [],
        'energy': []
    }
    tripletpotential = {
        'label': 'triplet',
        'scaling': None,
        'distance': [],
        'energy': []
    }
    so_coupling = {
        'label': 'so_coupling',
        'scaling': None,
        'distance': [],
        'energy': []
    }
    
    with open(file_path, 'r') as molscatoutput:
        start, midstart = False, False
        for line in molscatoutput:
            if "SHORT-RANGE POTENTIAL 1 SCALING FACTOR" in line:
                A_s = round(float(line.split()[9])*float(line.split()[11]), sigfigs = 12)   
                singletpotential['scaling'] = A_s
            elif "SHORT-RANGE POTENTIAL 2 SCALING FACTOR" in line:
                A_t = round(float(line.split()[9])*float(line.split()[11]), sigfigs = 12)   
                tripletpotential['scaling'] = A_t
            elif "SHORT-RANGE POTENTIAL 3 SCALING FACTOR" in line:
                A_so = round(float(line.split()[9])*float(line.split()[11]), sigfigs = 12)   
                so_coupling['scaling'] = A_so
            elif "PROPAGATION RANGES ARE CONTROLLED BY VARIABLES RMIN, RMID AND RMAX, WITH INPUT VALUES" in line:
                line = next(molscatoutput)
                rmin, rmid, rmax = round(float( line.split()[2] ), sigfigs = 8, warn = False), round(float( line.split()[5] ), sigfigs = 8, warn = False), round(float( line.split()[8] ), sigfigs = 8, warn = False)
            elif "PROPAGATION STEP SIZE DETERMINED USING DR" in line:
                dr = round(float(line.split()[7]), sigfigs = 8, warn = False)
            elif "P(I)" in line and not start and not midstart:
                start = True
                try:
                    r = rmin
                except NameError:
                    r = 0
                    warnings.warn("Minimum distance of propagation no specified")
            elif "P(I)" in line and start and not midstart:
                singletpotential['distance'].append( r )
                singletpotential['energy'].append( float(line.split()[2]) )
                tripletpotential['distance'].append( r )
                tripletpotential['energy'].append( float(line.split()[3]) )
                so_coupling['distance'].append( r )
                so_coupling['energy'].append( float(line.split()[4]) )
                r = round(r + dr/2, sigfigs = 8, warn = False)
            
            elif "MDPROP. LOG-DERIVATIVE MATRIX PROPAGATED FROM" in line:
                r -= dr/2
                if r != rmid:
                    print("r = ", r, "\n rmid = ", rmid)
                    warnings.warn("The short range propagation seems to end up not on rmax. Something's wrong.")
                line = next(molscatoutput)
                r += dr/2

            elif "CDIAG" in line:
                midstart = True
                r -= dr/2
                line = next(molscatoutput)
            
            elif "P(I)" in line and midstart:
                singletpotential['energy'].append( float(line.split()[2]) )
                tripletpotential['energy'].append( float(line.split()[3]) )
                so_coupling['energy'].append( float(line.split()[4]) )
                line = next(molscatoutput)
                singletpotential['energy'].append( float(line.split()[2]) )
                tripletpotential['energy'].append( float(line.split()[3]) )
                so_coupling['energy'].append( float(line.split()[4]) )
                line = next(molscatoutput)
                if "AIPROP" in line:
                    line = next(molscatoutput)
                dr = round(float(line.split()[2]), sigfigs = 8, warn = False)
                

                r = round(r + dr/2, sigfigs = 8, warn = False)
                singletpotential['distance'].append( r )
                tripletpotential['distance'].append( r )
                so_coupling['distance'].append( r )
                
                r = round(r + dr/2, sigfigs = 8, warn = False)
                singletpotential['distance'].append( r )
                tripletpotential['distance'].append( r )
                so_coupling['distance'].append( r )

    # get the true value used by molscat
    # so_coupling['energy'] = list(hartree_in_inv_cm*fine_structure_constant**2 * np.array(so_coupling['energy']))

    return singletpotential, tripletpotential, so_coupling


def main():
    parser_description = "This is a python script for collecting the potential and spin-orbit coupling curves fitted by RKHS from MOLSCAT outputs."
    parser = argparse.ArgumentParser(description=parser_description)
    parser.add_argument("-i", "--input", type = str, required = True, help = "Path to the input file or directory")
    parser.add_argument("-o", "--output", type = str, required = True, help = "Path to the output file or directory")
    # parser.add_argument("-r", "--recursive", action = 'store_true', help = "If enabled, the input directory will be searched for .output files recursively.")
    args = parser.parse_args()

    if Path(args.input).is_file() and not Path(args.output).is_dir():
        potential_data = get_potentials_and_so(args.input)
        update_json(potential_data, args.output)
    elif Path(args.input).is_dir():
        Path(args.output).mkdir(parents=True, exist_ok=True)
        for file_path in Path(args.input).iterdir():
            if file_path.is_file() and file_path.name.endswith('.output'):
                potential_data = get_potentials_and_so(file_path)
                update_json(potential_data, Path(args.output).joinpath(file_path.with_suffix('.json')).name)
                print(file_path.name, " read.")
    else:
        print("Input and output should both be .json files or both should be directories. Try again")
        return

if __name__ == "__main__":
    start_time = time.time()
    main()
    print("---Time of collecting the potentials was %s seconds ---" % (time.time() - start_time))