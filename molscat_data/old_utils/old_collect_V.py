import warnings

from scaling_old import update_json
from sigfig import round
import argparse
import os
from pathlib import Path
import time

# filepath = r"C:\Users\maksw\Documents\python\data\SO\RKHS\molscat-RbSr+30.output"


def get_potentials(filepath):

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
    
    with open(filepath, 'r') as molscatoutput:
        start, midstart = False, False
        for line in molscatoutput:
            if "SHORT-RANGE POTENTIAL 1 SCALING FACTOR" in line:
                A_s = round(float(line.split()[9])*float(line.split()[11]), sigfigs = 12)   
                singletpotential['scaling'] = A_s
            elif "SHORT-RANGE POTENTIAL 2 SCALING FACTOR" in line:
                A_t = round(float(line.split()[9])*float(line.split()[11]), sigfigs = 12)   
                tripletpotential['scaling'] = A_t
            
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
                    print("WARNING: minimum distance of propagation no specified")
            elif "P(I)" in line and start and not midstart:
                singletpotential['distance'].append( r )
                singletpotential['energy'].append( float(line.split()[2]) )
                tripletpotential['distance'].append( r )
                tripletpotential['energy'].append( float(line.split()[3]) )
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
                # singletpotential['distance'].append( r )
                singletpotential['energy'].append( float(line.split()[2]) )
                tripletpotential['energy'].append( float(line.split()[3]) )
                line = next(molscatoutput)
                singletpotential['energy'].append( float(line.split()[2]) )
                tripletpotential['energy'].append( float(line.split()[3]) )
                line = next(molscatoutput)
                if "AIPROP" in line:
                    line = next(molscatoutput)
                dr = round(float(line.split()[2]), sigfigs = 8, warn = False)
                

                r = round(r + dr/2, sigfigs = 8, warn = False)
                singletpotential['distance'].append( r )
                tripletpotential['distance'].append( r )
                
                r = round(r + dr/2, sigfigs = 8, warn = False)
                singletpotential['distance'].append( r )
                tripletpotential['distance'].append( r )
    return singletpotential, tripletpotential


def main():
    parser_description = "This is a python script for collecting the potential curves fitted by RKHS from MOLSCAT outputs."
    parser = argparse.ArgumentParser(description=parser_description)
    parser.add_argument("-i", "--input", type = str, required = True, help = "Path to the input file or directory")
    parser.add_argument("-o", "--output", type = str, required = True, help = "Path to the output file or directory")
    # parser.add_argument("-r", "--recursive", action = 'store_true', help = "If enabled, the input directory will be searched for .output files recursively.")
    args = parser.parse_args()

    if os.path.isfile(args.input) and not os.path.isdir(args.output):
        potential_data = get_potentials(args.input)
        update_json(potential_data, args.output)
    elif os.path.isdir(args.input):
        Path(args.output).mkdir(parents=True, exist_ok=True)
        for file in os.scandir(args.input):
            if file.is_file() and file.name.endswith('.output'):
                potential_data = get_potentials(file.path)
                update_json(potential_data, os.path.join(args.output,file.name.strip('.output')+r'.json'))
                print(file.name, " read.")
    else:
        print("Input and output should both be .json files or both should be directories. Try again")
        return

if __name__ == "__main__":
    start_time = time.time()
    main()
    print("---Time of collecting the potentials was %s seconds ---" % (time.time() - start_time))