import warnings

from pathlib import Path
import json

import numpy as np
import cmath

from scipy.interpolate import interp1d
from scipy import optimize

# from more_itertools import unique_everseen
from .physical_constants import red_mass_87Rb_88Sr_amu

default_singlet_scaling_path = Path(__file__).parents[2] / 'data' / 'scaling_old' / 'singlet_vs_coeff.json'
default_triplet_scaling_path = default_singlet_scaling_path.parent / 'triplet_vs_coeff.json'


class ComplexEncoder(json.JSONEncoder):
    def default(self, z):
        if isinstance(z, complex):
            return { '__complex__': True, 'abs': abs(z), 'phase': cmath.phase(z) }
        return json.JSONEncoder.default(self, z)

def decode_complex(dct):
    if '__complex__' in dct:
        return cmath.rect(dct['abs'], dct['phase'])
    return dct

def read_from_json(json_path: str | Path, recursive: bool = False) -> None:
    data = []
    # print(json_path, "--- input.")
    if not isinstance(json_path, Path):
        json_path = Path(json_path)
    
    if json_path.is_file() and json_path.suffix == '.json':
        with open(json_path) as json_file:
                data = json.load(json_file, object_hook=decode_complex)
    elif json_path.is_dir():
        for file in json_path.iterdir():
            if file.is_file() and file.suffix == '.json':
                data.extend(read_from_json(file, recursive = recursive))
                # print(file.path, "read.")
            if recursive and file.is_dir():
                data.extend(read_from_json(file, recursive = recursive))
    else:
        print("The input path is neither .json file nor a directory! Nothing can be done.")
    return data

def update_json(new_data, json_path):
    """
    Now update_json will create a .json file if no existing is found.
    
    """

    if not isinstance(json_path, Path):
        json_path = Path(json_path)

    if not json_path.suffix == '.json':
        warnings.warn("The file path doesn't end with '.json'.")

    if json_path.is_file():
        # Update and existing json file if it exist
        with open(json_path,'r+') as file:
            # First we load existing data into a dict.
            file_data = json.load(file)
            # Join new_data with file_data inside emp_details
            file_data += new_data
            # Sets file's current position at offset.
            file.seek(0)
            # convert back to json.
            json.dump(file_data, file, cls=ComplexEncoder, indent = 2)
            # print(filepath, "updated.")
    else:
        # Create a new json file if it doesn't exist
        # Create a parent directory if it doesn't exist
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, 'w') as output:
            # load the data into the .json file
            json.dump(new_data, output, cls=ComplexEncoder, indent = 2)


def scattering_length_from_phase(phase, C4 = -159.9, red_mass = red_mass_87Rb_88Sr_amu):
    """
    Returns the value of the scattering length as calculated from the semiclassical phase
    Phys. Rev. A 48, 546–553 (1993)
    """
    return -np.sqrt(2*red_mass*np.abs(C4)) * np.tan(phase - np.pi/4)

def phase_from_scattering_length(scattering_length, C4 = -159.9, red_mass = red_mass_87Rb_88Sr_amu):
    """
    Returns the value of the (semiclassical phase integral + pi/4) modulo pi as calculated from the scattering length
    Eq. (A.9) in the MSc thesis
    Phys. Rev. A 48, 546–553 (1993)
    """
    return np.pi/2 + np.arctan(-scattering_length/np.sqrt(2*red_mass*np.abs(C4)))

def semiclassical_phase_function(single_channel_data_path: str | Path):
    """
    Returns ( semiclassical phase integral + pi/4 ) modulo pi as a multiple of pi!
    """

    if Path(single_channel_data_path).is_file(): # and single_channel_data_path.endswith(r'.json'):
        data = read_from_json(single_channel_data_path)
        data =  sorted(data, key = lambda i: (i["potential_name"], i["coefficient"] ) )
        x, y = [], []
        for item in data:
            x.append(item['coefficient'])
            # y.append(phase_from_scattering_length(item['scattering_length']) / np.pi)
            y.append(item["scattering_phase/pi"])
        f = interp1d(x,y)
        return f
    else:
        print("Supplied %s file is not a .json file or doesn't exist. Try again:)" % single_channel_data_path)
        return None

def inverse_function(y, function, starting_point):
    fun = lambda x: function(x) - y
    inverse = optimize.newton(fun, starting_point)
    return inverse

def parameter_from_semiclassical_phase(phase, single_channel_data_path, starting_points):
    f = semiclassical_phase_function(single_channel_data_path)
    try:
        parameter = inverse_function(phase, f, starting_points[0])
    except ValueError:
        parameter = inverse_function(phase, f, starting_points[1])
    return parameter

default_singlet_phase_function = semiclassical_phase_function(default_singlet_scaling_path)

default_triplet_phase_function = semiclassical_phase_function(default_triplet_scaling_path)

def default_singlet_parameter_from_phase(phase):
    """Wrapper around parameter_from_semiclassical_phase."""
    return parameter_from_semiclassical_phase(phase, single_channel_data_path = default_singlet_scaling_path, starting_points = [1.000,1.010])

def default_triplet_parameter_from_phase(phase):
    """Wrapper around parameter_from_semiclassical_phase."""
    return parameter_from_semiclassical_phase(phase, single_channel_data_path = default_triplet_scaling_path, starting_points = [1.000,0.996])