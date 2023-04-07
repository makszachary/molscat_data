from pathlib import Path
import json

import numpy as np
import cmath

from scipy.interpolate import interp1d
from scipy import optimize

# from more_itertools import unique_everseen
from .physical_constants import red_mass_87Rb_88Sr

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

def scattering_length_from_phase(phase, C4 = -159.9, red_mass = red_mass_87Rb_88Sr):
    """
    Returns the value of the scattering length as calculated from the semiclassical phase
    Phys. Rev. A 48, 546–553 (1993)
    """
    return -np.sqrt(2*red_mass*np.abs(C4)) * np.tan(phase - np.pi/4)

def phase_from_scattering_length(scattering_length, C4 = -159.9, red_mass = red_mass_87Rb_88Sr):
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