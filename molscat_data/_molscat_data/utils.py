from typing import Any
import subprocess
import os
from pathlib import Path, PurePath
import re
import argparse

from multiprocessing import Pool
# from multiprocessing.dummy import Pool as ThreadPool 

import itertools

import numpy as np
from sigfig import round

import time

from _molscat_data.smatrix import SMatrix, SMatrixCollection, CollectionParameters, CollectionParametersIndices
from _molscat_data import quantum_numbers as qn
from _molscat_data.thermal_averaging import n_root_scale
from _molscat_data.scaling_old import parameter_from_semiclassical_phase, semiclassical_phase_function
from _molscat_data.effective_probability import effective_probability
from _molscat_data.physical_constants import amu_to_au


def rate_fmfsms(s_matrix_collection: SMatrixCollection, F_out: int, MF_out: int, S_out: int, MS_out: int, F_in: int, MF_in: int, S_in: int, MS_in: int, param_indices: dict = None, dLMax: int = 4, unit = None) -> float:
    L_max = max(key[0].L for s_matrix in s_matrix_collection.matrixCollection.values() for key in s_matrix.matrix.keys())
    rate = np.sum( [ s_matrix_collection.getRateCoefficient(qn.LF1F2(L = L_out, ML = ML_in + MF_in + MS_in - MF_out - MS_out, F1 = F_out, MF1 = MF_out, F2 = S_out, MF2 = MS_out), qn.LF1F2(L = L_in, ML = ML_in, F1 = F_in, MF1 = MF_in, F2 = S_out, MF2 = MS_in), unit = unit, param_indices = param_indices) for L_in in range(0, L_max+1, 2) for ML_in in range(-L_in, L_in+1, 2) for L_out in range(L_in - dLMax*2, L_in + dLMax*2+1, 2*2) if (L_out >= 0 and L_out <=L_max) ], axis = 0 )
    return rate

def rate_fmfsms_vs_L(s_matrix_collection: SMatrixCollection, F_out: int, MF_out: int, S_out: int, MS_out: int, F_in: int, MF_in: int, S_in: int, MS_in: int, param_indices: dict = None, dLMax: int = 4, unit = None) -> float:
    L_max = max(key[0].L for s_matrix in s_matrix_collection.matrixCollection.values() for key in s_matrix.matrix.keys())
    rate = np.array( [ s_matrix_collection.getRateCoefficient(qn.LF1F2(L = L_out, ML = ML_in + MF_in + MS_in - MF_out - MS_out, F1 = F_out, MF1 = MF_out, F2 = S_out, MF2 = MS_out), qn.LF1F2(L = L_in, ML = ML_in, F1 = F_in, MF1 = MF_in, F2 = S_out, MF2 = MS_in), unit = unit, param_indices = param_indices) for L_in in range(0, L_max+1, 2) for ML_in in range(-L_in, L_in+1, 2) for L_out in range(L_in - dLMax*2, L_in + dLMax*2+1, 2*2) if (L_out >= 0 and L_out <=L_max) ] )
    return rate

def rate_Lfmfsms(s_matrix_collection: SMatrixCollection, L_in: int, F_out: int, MF_out: int, S_out: int, MS_out: int, F_in: int, MF_in: int, S_in: int, MS_in: int, param_indices: dict = None, dLMax: int = 4, unit = None) -> float:
    L_max = max(key[0].L for s_matrix in s_matrix_collection.matrixCollection.values() for key in s_matrix.matrix.keys())
    rate = np.sum( [ s_matrix_collection.getRateCoefficient(qn.LF1F2(L = L_out, ML = ML_in + MF_in + MS_in - MF_out - MS_out, F1 = F_out, MF1 = MF_out, F2 = S_out, MF2 = MS_out), qn.LF1F2(L = L_in, ML = ML_in, F1 = F_in, MF1 = MF_in, F2 = S_out, MF2 = MS_in), unit = unit, param_indices = param_indices) for ML_in in range(-L_in, L_in+1, 2) for L_out in range(L_in - dLMax*2, L_in + dLMax*2+1, 2*2) if (L_out >= 0 and L_out <= L_max) ], axis = 0 )
    return rate

def rate_fmfsms_vs_L_SE(s_matrix_collection: SMatrixCollection, F_out: int, MF_out: int, S_out: int, MS_out: int, F_in: int, MF_in: int, S_in: int, MS_in: int, param_indices: dict = None, unit = None) -> float:
    L_max = max(key[0].L for s_matrix in s_matrix_collection.matrixCollection.values() for key in s_matrix.matrix.keys())
    rate = np.array( [ (L_in+1) * s_matrix_collection.getRateCoefficient(qn.LF1F2(L = L_in, ML = 0, F1 = F_out, MF1 = MF_out, F2 = S_out, MF2 = MS_out), qn.LF1F2(L = L_in, ML = 0, F1 = F_in, MF1 = MF_in, F2 = S_in, MF2 = MS_in), unit = unit, param_indices = param_indices) for L_in in range(0, L_max+1, 2)] )
    return rate

def rate_fmfsms_vs_L_multiprocessing(s_matrix_collection: SMatrixCollection, F_out: int, MF_out: int, S_out: int, MS_out: int, F_in: int, MF_in: int, S_in: int, MS_in: int, param_indices: dict = None, dLMax: int = 4, unit = None) -> float:
    
    args = locals().copy()
    args.pop('s_matrix_collection')
    args.pop('unit')
    args.pop('param_indices')
    args.pop('dLMax')

    L_max = max(key[0].L for s_matrix in s_matrix_collection.matrixCollection.values() for key in s_matrix.matrix.keys())

    with Pool() as pool:
        arguments = tuple( (s_matrix_collection, L_in, *(args[name] for name in args), param_indices, dLMax, unit) for L_in in range(0, L_max + 1, 2))
        print(arguments)
        results = pool.starmap(rate_Lfmfsms, arguments)
        rate = np.array(results)
        return rate


def probability(s_matrix_collection: SMatrixCollection, F_out: int | np.ndarray[Any, int], MF_out: int | np.ndarray[Any, int], S_out: int | np.ndarray[Any, int], MS_out: int | np.ndarray[Any, int], F_in: int | np.ndarray[Any, int], MF_in: int | np.ndarray[Any, int], S_in: int | np.ndarray[Any, int], MS_in: int | np.ndarray[Any, int], param_indices = None, dLMax: int = 4) -> np.ndarray[Any, float]:
    
    args = locals().copy()
    args.pop('s_matrix_collection')
    args.pop('param_indices')
    args.pop('dLMax')
    arg_shapes = tuple( value.shape for value in args.values() if isinstance(value, np.ndarray) )

    averaged_momentum_transfer_rate = s_matrix_collection.getThermallyAveragedMomentumTransferRate(qn.LF1F2(None, None, F1 = 2, MF1 = 2, F2 = 1, MF2 = -1), param_indices = param_indices)

    # convert all arguments to np.ndarrays if any of them is an instance np.ndarray
    array_like = False
    if any( isinstance(arg, np.ndarray) for arg in args.values() ):
        array_like = True
        arg_shapes = tuple( value.shape for value in args.values() if isinstance(value, np.ndarray) )
        if any(arg_shape != arg_shapes[0] for arg_shape in arg_shapes): raise ValueError(f"The shape of the numpy arrays passed as arguments should be the same.")
        
        for name, arg in args.items():
            if not isinstance(arg, np.ndarray):
                args[name] = np.full(arg_shapes[0], arg)


    if array_like:
        with Pool() as pool:
           arguments = ( (s_matrix_collection, *(args[name][index] for name in args), param_indices, dLMax) for index in np.ndindex(arg_shapes[0]))
           results = pool.starmap(rate_fmfsms, arguments)
           rate_shape = results[0].shape
           rate = np.array(results).reshape((*arg_shapes[0], *rate_shape))

           averaged_rate = s_matrix_collection.thermalAverage(rate)
           averaged_momentum_transfer_rate = np.full_like(averaged_rate, averaged_momentum_transfer_rate)
           probability = averaged_rate / averaged_momentum_transfer_rate

           return probability
    
    rate = rate_fmfsms(s_matrix_collection, **args)
    averaged_rate = s_matrix_collection.thermalAverage(rate)
    probability = averaged_rate / averaged_momentum_transfer_rate

    return probability