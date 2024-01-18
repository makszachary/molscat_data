from typing import Any
import subprocess
import os
import sys
from pathlib import Path, PurePath
import re
import argparse
import copy

from multiprocessing import Pool
import multiprocessing
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
    t0 = time.perf_counter()
    L_max = max(key[0].L for s_matrix in s_matrix_collection.matrixCollection.values() for key in s_matrix.matrix.keys())
    rate = np.sum( [ s_matrix_collection.getRateCoefficient(qn.LF1F2(L = L_out, ML = ML_in + MF_in + MS_in - MF_out - MS_out, F1 = F_out, MF1 = MF_out, F2 = S_out, MF2 = MS_out), qn.LF1F2(L = L_in, ML = ML_in, F1 = F_in, MF1 = MF_in, F2 = S_in, MF2 = MS_in), unit = unit, param_indices = param_indices) for L_in in range(0, L_max+1, 2) for ML_in in range(-L_in, L_in+1, 2) for L_out in range(L_in - dLMax*2, L_in + dLMax*2+1, 2*2) if (L_out >= 0 and L_out <=L_max) ], axis = 0 )
    print(rate, f'The time of calculating the rate was {(time.perf_counter()-t0):.2f} s.')
    return rate

def rate_fmfsms_vs_L(s_matrix_collection: SMatrixCollection, F_out: int, MF_out: int, S_out: int, MS_out: int, F_in: int, MF_in: int, S_in: int, MS_in: int, param_indices: dict = None, dLMax: int = 4, unit = None) -> float:
    args = locals().copy()
    args.pop('s_matrix_collection')
    args.pop('param_indices')
    args.pop('dLMax')
    args.pop('unit')
    t0 = time.perf_counter()
    L_max = max(key[0].L for s_matrix in s_matrix_collection.matrixCollection.values() for key in s_matrix.matrix.keys())
    print(f"Starting  the calculations for a single combination of quantum numbers ({args}).", flush = True)
    shp = s_matrix_collection.getRateCoefficient(qn.LF1F2(L = 0, ML = 0 + MF_in + MS_in - MF_out - MS_out, F1 = F_out, MF1 = MF_out, F2 = S_out, MF2 = MS_out), qn.LF1F2(L = 0, ML = 0, F1 = F_in, MF1 = MF_in, F2 = S_in, MF2 = MS_in), unit = unit, param_indices = param_indices).shape
    try:
        rate = np.array( [np.sum([ s_matrix_collection.getRateCoefficient(qn.LF1F2(L = L_out, ML = ML_in + MF_in + MS_in - MF_out - MS_out, F1 = F_out, MF1 = MF_out, F2 = S_out, MF2 = MS_out), qn.LF1F2(L = L_in, ML = ML_in, F1 = F_in, MF1 = MF_in, F2 = S_in, MF2 = MS_in), unit = unit, param_indices = param_indices) if (L_out >= 0 and L_out <=L_max) and abs(ML_in + MF_in + MS_in - MF_out - MS_out) <= L_out else np.zeros(shp) for ML_in in range(-L_in, L_in+1, 2) for L_out in range(L_in - dLMax*2, L_in + dLMax*2+1, 2*2) ], axis = 0) for L_in in range(0, L_max+1, 2)])
        print(f'{L_max =}')
        print(f'{rate.shape =}', flush = True)
    except ValueError as err:
        try:
            err_shapes = [np.sum([ s_matrix_collection.getRateCoefficient(qn.LF1F2(L = L_out, ML = ML_in + MF_in + MS_in - MF_out - MS_out, F1 = F_out, MF1 = MF_out, F2 = S_out, MF2 = MS_out), qn.LF1F2(L = L_in, ML = ML_in, F1 = F_in, MF1 = MF_in, F2 = S_in, MF2 = MS_in), unit = unit, param_indices = param_indices) for ML_in in range(-L_in, L_in+1, 2) for L_out in range(L_in - dLMax*2, L_in + dLMax*2+1, 2*2) if (L_out >= 0 and L_out <=L_max) and abs(ML_in + MF_in + MS_in - MF_out - MS_out) <= L_out ], axis = 0).shape for L_in in range(0, L_max+1, 2)]
            err_rate = [np.sum([ s_matrix_collection.getRateCoefficient(qn.LF1F2(L = L_out, ML = ML_in + MF_in + MS_in - MF_out - MS_out, F1 = F_out, MF1 = MF_out, F2 = S_out, MF2 = MS_out), qn.LF1F2(L = L_in, ML = ML_in, F1 = F_in, MF1 = MF_in, F2 = S_in, MF2 = MS_in), unit = unit, param_indices = param_indices) for ML_in in range(-L_in, L_in+1, 2) for L_out in range(L_in - dLMax*2, L_in + dLMax*2+1, 2*2) if (L_out >= 0 and L_out <=L_max) and abs(ML_in + MF_in + MS_in - MF_out - MS_out) <= L_out ], axis = 0) for L_in in range(0, L_max+1, 2)]
            raise ValueError(f'An error occurred in rate_fmfsms_vs_L function for {args}.\n{err}\n{err_rate =}\n{err_shapes =}')
        except ValueError as new_err:
            raise ValueError(f'An error occurred in rate_fmfsms_vs_L function for {args}. \n{new_err =}')
    # rate = np.array( [np.sum([ np.full(shape, 1) for ML_in in range(-L_in, L_in+1, 2) for L_out in range(L_in - dLMax*2, L_in + dLMax*2+1, 2*2) if (L_out >= 0 and L_out <=L_max) and abs(ML_in + MF_in + MS_in - MF_out - MS_out) <= L_out ], axis = 0) for L_in in range(0, L_max+1, 2)])
    # print(f'{rate.shape =}', flush=True)
    print(f"The time of the calculations for a single combination of quantum numbers ({args}) was {time.perf_counter()-t0:.2f} s.", flush = True)
    return rate

def rate_Lfmfsms(s_matrix_collection: SMatrixCollection, L_in: int, F_out: int, MF_out: int, S_out: int, MS_out: int, F_in: int, MF_in: int, S_in: int, MS_in: int, param_indices: dict = None, param_values: dict = None, dLMax: int = 4, unit = None) -> float:
    L_max = max(key[0].L for s_matrix in s_matrix_collection.matrixCollection.values() for key in s_matrix.matrix.keys())
    rate = np.sum( [ s_matrix_collection.getRateCoefficient(qn.LF1F2(L = L_out, ML = ML_in + MF_in + MS_in - MF_out - MS_out, F1 = F_out, MF1 = MF_out, F2 = S_out, MF2 = MS_out), qn.LF1F2(L = L_in, ML = ML_in, F1 = F_in, MF1 = MF_in, F2 = S_in, MF2 = MS_in), unit = unit, param_indices = param_indices, param_values = param_values) for ML_in in range(-L_in, L_in+1, 2) for L_out in range(L_in - dLMax*2, L_in + dLMax*2+1, 2*2) if (L_out >= 0 and L_out <= L_max) ], axis = 0 )
    return rate

def rate_fmfsms_vs_L_SE(s_matrix_collection: SMatrixCollection, F_out: int, MF_out: int, S_out: int, MS_out: int, F_in: int, MF_in: int, S_in: int, MS_in: int, param_indices: dict = None, unit = None) -> float:
    L_max = max(key[0].L for s_matrix in s_matrix_collection.matrixCollection.values() for key in s_matrix.matrix.keys())
    # rate = np.array( [ (L_in+1) * s_matrix_collection.getRateCoefficient(qn.LF1F2(L = L_in, ML = 0, F1 = F_out, MF1 = MF_out, F2 = S_out, MF2 = MS_out), qn.LF1F2(L = L_in, ML = 0, F1 = F_in, MF1 = MF_in, F2 = S_in, MF2 = MS_in), unit = unit, param_indices = param_indices) if (MF_in + MS_in == MF_out + MS_out) else 0 for L_in in range(0, L_max+1, 2)] )
    rate = np.array( [ (L_in+1) * s_matrix_collection.getRateCoefficient(qn.LF1F2(L = L_in, ML = 0, F1 = F_out, MF1 = MF_out, F2 = S_out, MF2 = MS_out), qn.LF1F2(L = L_in, ML = 0, F1 = F_in, MF1 = MF_in, F2 = S_in, MF2 = MS_in), unit = unit, param_indices = param_indices) for L_in in range(0, L_max+1, 2)] )
    return rate

def rate_fmfsms_vs_L_multiprocessing(s_matrix_collection: SMatrixCollection, F_out: int, MF_out: int, S_out: int, MS_out: int, F_in: int, MF_in: int, S_in: int, MS_in: int, param_indices: dict = None, param_values: dict = None, dLMax: int = 4, unit = None) -> float:
    
    args = locals().copy()
    args.pop('s_matrix_collection')
    args.pop('unit')
    args.pop('param_indices')
    args.pop('param_values')
    args.pop('dLMax')

    L_max = max(key[0].L for s_matrix in s_matrix_collection.matrixCollection.values() for key in s_matrix.matrix.keys())

    with multiprocessing.get_context('spawn').Pool() as pool:
        arguments = tuple( (s_matrix_collection, L_in, *(args[name] for name in args), param_indices, param_values, dLMax, unit) for L_in in range(0, L_max + 1, 2))
        results = pool.starmap(rate_Lfmfsms, arguments)
        rate = np.array(results)
        return rate


def probability(s_matrix_collection: SMatrixCollection, F_out: int | np.ndarray[Any, int], MF_out: int | np.ndarray[Any, int], S_out: int | np.ndarray[Any, int], MS_out: int | np.ndarray[Any, int], F_in: int | np.ndarray[Any, int], MF_in: int | np.ndarray[Any, int], S_in: int | np.ndarray[Any, int], MS_in: int | np.ndarray[Any, int], param_indices = None, dLMax: int = 4) -> np.ndarray[Any, float]:
    
    args = locals().copy()
    args.pop('s_matrix_collection')
    args.pop('param_indices')
    args.pop('dLMax')
    arg_shapes = tuple( value.shape for value in args.values() if isinstance(value, np.ndarray) )

    averaged_momentum_transfer_rate = s_matrix_collection.getThermallyAveragedMomentumTransferRate(qn.LF1F2(None, None, F1 = 2, MF1 = -2, F2 = 1, MF2 = -1), param_indices = param_indices)

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
        with multiprocessing.get_context('spawn').Pool() as pool:
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


def probability_not_parallel(s_matrix_collection: SMatrixCollection, F_out: int | np.ndarray[Any, int], MF_out: int | np.ndarray[Any, int], S_out: int | np.ndarray[Any, int], MS_out: int | np.ndarray[Any, int], F_in: int | np.ndarray[Any, int], MF_in: int | np.ndarray[Any, int], S_in: int | np.ndarray[Any, int], MS_in: int | np.ndarray[Any, int], param_indices = None, dLMax: int = 4) -> np.ndarray[Any, float]:
    
    args = locals().copy()
    args.pop('s_matrix_collection')
    args.pop('param_indices')
    args.pop('dLMax')
    arg_shapes = tuple( value.shape for value in args.values() if isinstance(value, np.ndarray) )

    t0=time.perf_counter()
    averaged_momentum_transfer_rate = s_matrix_collection.getThermallyAveragedMomentumTransferRate(qn.LF1F2(None, None, F1 = 2, MF1 = -2, F2 = 1, MF2 = -1), param_indices = param_indices)
    print(f'{averaged_momentum_transfer_rate.shape=}, the time of calculation was {time.perf_counter()-t0:.2f} s.')
    # print(f'{averaged_momentum_transfer_rate.shape=}')

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
        arguments = tuple( (s_matrix_collection, *(args[name][index] for name in args), param_indices, dLMax) for index in np.ndindex(arg_shapes[0]))
        results = [ rate_fmfsms(*arg) for arg in arguments ]
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


def k_L_E_parallel(s_matrix_collection: SMatrixCollection, F_out: int | np.ndarray[Any, int], MF_out: int | np.ndarray[Any, int], S_out: int | np.ndarray[Any, int], MS_out: int | np.ndarray[Any, int], F_in: int | np.ndarray[Any, int], MF_in: int | np.ndarray[Any, int], S_in: int | np.ndarray[Any, int], MS_in: int | np.ndarray[Any, int], param_indices = None, dLMax: int = 4, pc = False) -> np.ndarray[Any, float]:
    
    args = locals().copy()
    args.pop('s_matrix_collection')
    args.pop('param_indices')
    args.pop('dLMax')
    args.pop('pc')
    arg_shapes = tuple( value.shape for value in args.values() if isinstance(value, np.ndarray) )

    t0=time.perf_counter()
    momentum_transfer_rate = s_matrix_collection.getMomentumTransferRateCoefficientVsL(qn.LF1F2(None, None, F1 = 2, MF1 = -2, F2 = 1, MF2 = -1), unit = 'cm**3/s', param_indices = param_indices)
    print(f'{momentum_transfer_rate.shape=}, the time of calculation was {time.perf_counter()-t0:.2f} s.')

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
        if sys.platform == 'win32' or pc:
            ncores = multiprocessing.cpu_count()
        else:
            try:
                ncores = int(os.environ['SLURM_NTASKS_PER_NODE'])
            except KeyError:
                ncores = 1
            try:
                ncores *= int(os.environ['SLURM_CPUS_PER_TASK'])
            except KeyError:
                ncores *= 1
        print(f'{ncores=}')
        print(f'Number of input/output state combinations to calculate = {args["F_out"].size}.')
        with multiprocessing.get_context('spawn').Pool(ncores) as pool:
            arguments = tuple( (copy.deepcopy(s_matrix_collection), *(args[name][index] for name in args), param_indices, dLMax, 'cm**3/s') for index in np.ndindex(arg_shapes[0]))
            results = pool.starmap(rate_fmfsms_vs_L, arguments)
            # results = pool.starmap_async(rate_fmfsms_vs_L, arguments)
            # results = np.array(results.get())
            rate_shape = results[0].shape
            rate = np.array(results).reshape((*arg_shapes[0], *rate_shape))
            momentum_transfer_rate = np.full((*arg_shapes[0], *momentum_transfer_rate.shape), momentum_transfer_rate)

            return rate, momentum_transfer_rate
    
    rate = rate_fmfsms_vs_L(s_matrix_collection, **args)

    return rate, momentum_transfer_rate


def k_L_E_parallel_fmf(s_matrix_collection: SMatrixCollection, F_out: int | np.ndarray[Any, int], MF_out: int | np.ndarray[Any, int], S_out: int | np.ndarray[Any, int], MS_out: int | np.ndarray[Any, int], F_in: int | np.ndarray[Any, int], MF_in: int | np.ndarray[Any, int], S_in: int | np.ndarray[Any, int], MS_in: int | np.ndarray[Any, int], param_indices = None, dLMax: int = 4) -> np.ndarray[Any, float]:
    
    args = locals().copy()
    args.pop('s_matrix_collection')
    args.pop('param_indices')
    args.pop('dLMax')
    arg_shapes = tuple( value.shape for value in args.values() if isinstance(value, np.ndarray) )

    t0=time.perf_counter()

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
        try:
            ncores = int(os.environ['SLURM_NTASKS_PER_NODE'])
        except KeyError:
            ncores = 1
        try:
            ncores *= int(os.environ['SLURM_CPUS_PER_TASK'])
        except KeyError:
            ncores *= 1
        print(f'{ncores=}')
        print(f'Number of input/output state combinations to calculate = {args["F_out"].size}.')
        with multiprocessing.get_context('spawn').Pool(ncores) as pool:
            arguments = tuple( (s_matrix_collection, *(args[name][index] for name in args), param_indices, dLMax, 'cm**3/s') for index in np.ndindex(arg_shapes[0]))
            results = pool.starmap(rate_fmfsms_vs_L, arguments)
            rate_shape = results[0].shape
            rate = np.array(results).reshape((*arg_shapes[0], *rate_shape))

            return rate
    
    rate = rate_fmfsms_vs_L(s_matrix_collection, **args)

    return rate


def k_L_E_parallel_fmf_SE(s_matrix_collection: SMatrixCollection, F_out: int | np.ndarray[Any, int], MF_out: int | np.ndarray[Any, int], S_out: int | np.ndarray[Any, int], MS_out: int | np.ndarray[Any, int], F_in: int | np.ndarray[Any, int], MF_in: int | np.ndarray[Any, int], S_in: int | np.ndarray[Any, int], MS_in: int | np.ndarray[Any, int], param_indices = None,) -> np.ndarray[Any, float]:
    
    args = locals().copy()
    args.pop('s_matrix_collection')
    args.pop('param_indices')
    arg_shapes = tuple( value.shape for value in args.values() if isinstance(value, np.ndarray) )

    t0=time.perf_counter()

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
        try:
            ncores = int(os.environ['SLURM_NTASKS_PER_NODE'])
        except KeyError:
            ncores = 1
            try:
                ncores *= int(os.environ['SLURM_CPUS_PER_TASK'])
            except KeyError:
                ncores *= 1
        print(f'{ncores=}')
        print(f'Number of input/output state combinations to calculate = {args["F_out"].size}.')
        with multiprocessing.get_context('spawn').Pool(ncores) as pool:
            arguments = tuple( (s_matrix_collection, *(args[name][index] for name in args), param_indices, 'cm**3/s') for index in np.ndindex(arg_shapes[0]))
            results = pool.starmap(rate_fmfsms_vs_L_SE, arguments)
            rate_shape = results[0].shape
            rate = np.array(results).reshape((*arg_shapes[0], *rate_shape))

            return rate
    
    rate = rate_fmfsms_vs_L_SE(s_matrix_collection, **args)

    return rate


def k_L_E_parallel_odd(s_matrix_collection: SMatrixCollection, F_out: int | np.ndarray[Any, int], MF_out: int | np.ndarray[Any, int], S_out: int | np.ndarray[Any, int], MS_out: int | np.ndarray[Any, int], F_in: int | np.ndarray[Any, int], MF_in: int | np.ndarray[Any, int], S_in: int | np.ndarray[Any, int], MS_in: int | np.ndarray[Any, int], param_indices = None, dLMax: int = 4) -> np.ndarray[Any, float]:
    
    args = locals().copy()
    args.pop('s_matrix_collection')
    args.pop('param_indices')
    args.pop('dLMax')
    arg_shapes = tuple( value.shape for value in args.values() if isinstance(value, np.ndarray) )

    t0=time.perf_counter()
    momentum_transfer_rate = s_matrix_collection.getMomentumTransferRateCoefficientVsL(qn.LF1F2(None, None, F1 = 2, MF1 = -2, F2 = 10, MF2 = -10), unit = 'cm**3/s', param_indices = param_indices)
    print(f'{momentum_transfer_rate.shape=}, the time of calculation was {time.perf_counter()-t0:.2f} s.')

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
        try:
            ncores = int(os.environ['SLURM_NTASKS_PER_NODE'])
        except KeyError:
            ncores = 1
        try:
            ncores *= int(os.environ['SLURM_CPUS_PER_TASK'])
        except KeyError:
            ncores *= 1
        print(f'{ncores=}')
        print(f'Number of input/output state combinations to calculate = {args["F_out"].size}.')

        with multiprocessing.get_context("spawn").Pool(ncores) as pool:
            arguments = tuple( (s_matrix_collection, *(args[name][index] for name in args), param_indices, dLMax, 'cm**3/s') for index in np.ndindex(arg_shapes[0]))
            results = pool.starmap(rate_fmfsms_vs_L, arguments)
            rate_shape = results[0].shape
            print(f'{rate_shape =}')
            rate = np.array(results).reshape((*arg_shapes[0], *rate_shape))
            momentum_transfer_rate = np.full((*arg_shapes[0], *momentum_transfer_rate.shape), momentum_transfer_rate)

            return rate, momentum_transfer_rate
    
    rate = rate_fmfsms_vs_L(s_matrix_collection, **args)

    return rate, momentum_transfer_rate



def k_L_E_not_parallel(s_matrix_collection: SMatrixCollection, F_out: int | np.ndarray[Any, int], MF_out: int | np.ndarray[Any, int], S_out: int | np.ndarray[Any, int], MS_out: int | np.ndarray[Any, int], F_in: int | np.ndarray[Any, int], MF_in: int | np.ndarray[Any, int], S_in: int | np.ndarray[Any, int], MS_in: int | np.ndarray[Any, int], param_indices = None, dLMax: int = 4) -> np.ndarray[Any, float]:    
    args = locals().copy()
    args.pop('s_matrix_collection')
    args.pop('param_indices')
    args.pop('dLMax')
    arg_shapes = tuple( value.shape for value in args.values() if isinstance(value, np.ndarray) )

    t0=time.perf_counter()
    momentum_transfer_rate = s_matrix_collection.getMomentumTransferRateCoefficientVsL(qn.LF1F2(None, None, F1 = 2, MF1 = -2, F2 = 1, MF2 = -1), unit = 'cm**3/s', param_indices = param_indices)

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
        arguments = tuple( (s_matrix_collection, *(args[name][index] for name in args), param_indices, dLMax, 'cm**3/s') for index in np.ndindex(arg_shapes[0]))
        results = [ rate_fmfsms_vs_L(*arg) for arg in arguments ]
        rate_shape = results[0].shape
        rate = np.array(results).reshape((*arg_shapes[0], *rate_shape))
        momentum_transfer_rate = np.full((*arg_shapes[0], *momentum_transfer_rate.shape), momentum_transfer_rate)

        return rate, momentum_transfer_rate
    
    rate = rate_fmfsms_vs_L(s_matrix_collection, **args)

    return rate, momentum_transfer_rate

def k_L_E_SE_not_parallel(s_matrix_collection: SMatrixCollection, F_out: int | np.ndarray[Any, int], MF_out: int | np.ndarray[Any, int], S_out: int | np.ndarray[Any, int], MS_out: int | np.ndarray[Any, int], F_in: int | np.ndarray[Any, int], MF_in: int | np.ndarray[Any, int], S_in: int | np.ndarray[Any, int], MS_in: int | np.ndarray[Any, int], param_indices = None) -> np.ndarray[Any, float]:
    args = locals().copy()
    args.pop('s_matrix_collection')
    args.pop('param_indices')
    arg_shapes = tuple( value.shape for value in args.values() if isinstance(value, np.ndarray) )

    t0=time.perf_counter()
    momentum_transfer_rate = s_matrix_collection.getMomentumTransferRateCoefficientVsL(qn.LF1F2(None, None, F1 = 4, MF1 = 4, F2 = 1, MF2 = 1), unit = 'cm**3/s', param_indices = param_indices)
    # print(momentum_transfer_rate)
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
        arguments = tuple( (s_matrix_collection, *(args[name][index] for name in args), param_indices, 'cm**3/s') for index in np.ndindex(arg_shapes[0]))
        results = [ rate_fmfsms_vs_L_SE(*arg) for arg in arguments ]
        rate_shape = results[0].shape
        rate = np.array(results).reshape((*arg_shapes[0], *rate_shape))
        momentum_transfer_rate = np.full((*arg_shapes[0], *momentum_transfer_rate.shape), momentum_transfer_rate)

        return rate, momentum_transfer_rate
    
    rate = rate_fmfsms_vs_L(s_matrix_collection, **args)

    return rate, momentum_transfer_rate