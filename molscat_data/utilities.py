import subprocess
import os
import re

import numpy as np

from smatrix import SMatrix, SMatrixCollection, CollectionParameters, CollectionParametersIndices
import quantum_numbers
from thermal_averaging import n_root_scale

molscat_executable_path = r'/net/people/plgrid/plgwalewski/molscat-RKHS-tcpld/molscat-exe/molscat-RKHS-tcpld'
molscat_input_template_path = r'../molscat/inputs/molscat-RbSr+.input'
molscat_input_path = r'../molscat/inputs/molscat-RbSr+filled.input'
molscat_output_path  = r'../molscat/outputs/molscat-RbSr+test.output'

E_min, E_max, nenergies, n = 4e-7, 4e-3, 100, 3

energyarray_str = str([round(n_root_scale(i, E_min, E_max, nenergies, n = n), sigfigs = 11) for i in range(nenergies+1)]).strip(']').strip('[')


with open(molscat_input_template_path, 'r') as molscat_template:
    input_content = molscat_template.read()
    input_content = re.sub("NENERGIES", str(int(nenergies)), input_content, flags = re.M)
    input_content = re.sub("ENERGYARRAY", energyarray_str, input_content, flags = re.M)

    with open(molscat_input_path, 'w') as molscat_input:
        molscat_input.write(input_content)
        molscat_input.truncate()            