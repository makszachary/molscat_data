import cProfile
import pstats
import multiprocessing
from typing import Any
import time
import sys
import os
from pathlib import Path
import numpy as np
import itertools

from sigfig import round

from _molscat_data.smatrix import SMatrixCollection
from _molscat_data import quantum_numbers as qn
from _molscat_data.thermal_averaging import n_root_scale, n_root_distribution, n_root_iterator
from _molscat_data.scaling_old import parameter_from_semiclassical_phase, default_singlet_phase_function, default_triplet_phase_function, default_singlet_parameter_from_phase, default_triplet_parameter_from_phase
# from _molscat_data.utils import k_L_E_parallel, k_L_E_not_parallel, k_L_E_parallel_from_path

PC = True
scratch_path = Path(__file__).parents[3] if PC else Path(os.path.expandvars('$SCRATCH'))

data_dir_path = Path(__file__).parents[1] / 'data'
outputs_dir_path = scratch_path / 'molscat' / 'outputs'
outputs_dir_path.mkdir(parents=True, exist_ok=True)
pickles_dir_path = scratch_path / 'python' / 'molscat_data' / 'data_produced' / 'pickles'
pickles_dir_path.mkdir(parents=True, exist_ok=True)
arrays_dir_path = pickles_dir_path.parent / 'arrays'
arrays_dir_path.mkdir(parents=True, exist_ok=True)
plots_dir_path = scratch_path / 'python' / 'molscat_data' / 'plots'


def mul(a, b):
    t0 = time.perf_counter()
    print(f'{a=}, {b=}')
    for i in range(100000000):
        j = i**2
    print(f"The time of a single computation for {a=}, {b=} was {time.perf_counter()-t0:.2f} s.", flush = True)
    return a * b

def measure_mp_and_lst_mul(args: tuple[tuple[float, float], ...], pc = False):
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
    
    t0 = time.perf_counter()
    results_lst = [ mul(*arg) for arg in args ]
    time_lst = time.perf_counter()-t0
    print(f"The time of list computations was {time_lst:.2f} s.")

    with multiprocessing.get_context('fork').Pool(ncores) as pool:
        t0 = time.perf_counter()
        results_mp = pool.starmap(mul, args)
        time_mp = time.perf_counter()-t0
        print(f"The time of multiprocessing computations was {time_mp:.2f} s.")
    
    return time_mp, time_lst

def collect_and_pickle(molscat_output_directory_path: Path | str, singlet_phase, triplet_phase, spinOrbitParameter: float | tuple[float, ...], energy_tuple: tuple[float, ...] ) -> tuple[SMatrixCollection, float, Path, Path]:
    time_0 = time.perf_counter()
    molscat_out_dir = scratch_path.joinpath('molscat', 'outputs')

    singlet_parameter = default_singlet_parameter_from_phase(singlet_phase)
    triplet_parameter = default_triplet_parameter_from_phase(triplet_phase)
    s_matrix_collection = SMatrixCollection(singletParameter = (singlet_parameter,), tripletParameter = (triplet_parameter,), collisionEnergy = energy_tuple)
    
    for output_path in Path(molscat_output_directory_path).iterdir():
        s_matrix_collection.update_from_output(file_path = output_path, non_molscat_so_parameter = spinOrbitParameter)
    
    pickle_path = pickles_dir_path / molscat_output_directory_path.relative_to(molscat_out_dir)
    pickle_path = pickle_path.parent / (pickle_path.name + '.pickle')
    pickle_path.parent.mkdir(parents = True, exist_ok = True)

    s_matrix_collection.toPickle(pickle_path)

    duration = time.perf_counter()-time_0

    return s_matrix_collection, duration, molscat_output_directory_path, pickle_path

def rate_fmfsms_vs_L(s_matrix_collection: SMatrixCollection, F_out: int, MF_out: int, S_out: int, MS_out: int, F_in: int, MF_in: int, S_in: int, MS_in: int, param_indices: dict = None, dLMax: int = 4, unit = None) -> float:
    args = locals().copy()
    args.pop('s_matrix_collection')
    args.pop('param_indices')
    args.pop('dLMax')
    args.pop('unit')
    t0 = time.perf_counter()
    L_max = max(key[0].L for s_matrix in s_matrix_collection.matrixCollection.values() for key in s_matrix.matrix.keys())
    print(f"Starting  the calculations for a single combination of quantum numbers ({args}).", flush = True)
    shape = s_matrix_collection.getRateCoefficient(qn.LF1F2(L = 0, ML = 0 + MF_in + MS_in - MF_out - MS_out, F1 = F_out, MF1 = MF_out, F2 = S_out, MF2 = MS_out), qn.LF1F2(L = 0, ML = 0, F1 = F_in, MF1 = MF_in, F2 = S_in, MF2 = MS_in), unit = unit, param_indices = param_indices).shape
    rate = np.array( [np.sum([ s_matrix_collection.getRateCoefficient(qn.LF1F2(L = L_out, ML = ML_in + MF_in + MS_in - MF_out - MS_out, F1 = F_out, MF1 = MF_out, F2 = S_out, MF2 = MS_out), qn.LF1F2(L = L_in, ML = ML_in, F1 = F_in, MF1 = MF_in, F2 = S_in, MF2 = MS_in), unit = unit, param_indices = param_indices) for ML_in in range(-L_in, L_in+1, 2) for L_out in range(L_in - dLMax*2, L_in + dLMax*2+1, 2*2) if (L_out >= 0 and L_out <=L_max) and abs(ML_in + MF_in + MS_in - MF_out - MS_out) <= L_out ], axis = 0) for L_in in range(0, L_max+1, 2)])
    # rate = np.array( [np.sum([ np.full(shape, 1) for ML_in in range(-L_in, L_in+1, 2) for L_out in range(L_in - dLMax*2, L_in + dLMax*2+1, 2*2) if (L_out >= 0 and L_out <=L_max) and abs(ML_in + MF_in + MS_in - MF_out - MS_out) <= L_out ], axis = 0) for L_in in range(0, L_max+1, 2)])
    print(f"The time of the calculations for a single combination of quantum numbers ({args}) was {time.perf_counter()-t0:.2f} s.", flush = True)
    return rate

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
<<<<<<< HEAD
        with multiprocessing.get_context('fork').Pool(ncores) as pool:
=======
        with multiprocessing.get_context('spawn').Pool(ncores) as pool:
>>>>>>> e753092d4186af391fb1a82934c894974d250af6
            arguments = tuple( (s_matrix_collection, *(args[name][index] for name in args), param_indices, dLMax, 'cm**3/s') for index in np.ndindex(arg_shapes[0]))
            results = pool.starmap(rate_fmfsms_vs_L, arguments)
            # results = pool.starmap_async(rate_fmfsms_vs_L, arguments)
            # results = np.array(results.get())
            rate_shape = results[0].shape
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


def measure_mp_and_lst_k_L_E(pickle_path: Path | str, phases: tuple[float, float], pc = False):
    s_matrix_collection = SMatrixCollection.fromPickle(pickle_path)

    param_indices = { "singletParameter": (s_matrix_collection.singletParameter.index(default_singlet_parameter_from_phase(phases[0])),), "tripletParameter": (s_matrix_collection.tripletParameter.index( default_triplet_parameter_from_phase(phases[1]) ), ) } if phases is not None else None
    
    dLmax = 4
    F_out, F_in, S = 2, 4, 1
    # MF_out, MS_out, MF_in, MS_in = np.meshgrid(np.arange(-F_out, F_out+1, 2), np.arange(-S, S+1, 2), np.arange(-F_in, F_in+1, 2), S, indexing = 'ij')
    MF_out, MS_out, MF_in, MS_in = np.meshgrid(np.arange(-F_out, F_out+1, 2), np.arange(-S, S+1, 2), -F_in, S, indexing = 'ij')
    arg_hpf_deexcitation = (s_matrix_collection, F_out, MF_out, S, MS_out, F_in, MF_in, S, MS_in, param_indices, dLmax)
    # arg_hpf_deexcitation = (pickle_path, F_out, MF_out, S, MS_out, F_in, MF_in, S, MS_in, param_indices, dLmax)

    t0 = time.perf_counter()
    _, __ = k_L_E_not_parallel(*arg_hpf_deexcitation)
    time_lst = time.perf_counter() - t0
    print(f"Time of sequential calculations was {time_lst:.2f} s.")

    t0 = time.perf_counter()
    _, __ = k_L_E_parallel(*arg_hpf_deexcitation, pc = pc)
    time_mp = time.perf_counter() - t0
    print(f"Time of multiprocessing calculations was {time_mp:.2f} s.")

    return time_mp, time_lst


def main():
    print(sys.platform)
<<<<<<< HEAD
    # args = tuple((a, b) for a in range(4) for b in range(4))
    # time_mp_mul, time_lst_mul = measure_mp_and_lst_mul(args, pc = PC)
=======
    args = tuple((a, b) for a in range(4) for b in range(4))
    # with cProfile.Profile() as pr_simple:
    #     time_mp_mul, time_lst_mul = measure_mp_and_lst_mul(args, pc = PC)
>>>>>>> e753092d4186af391fb1a82934c894974d250af6
    
    # stats1 = pstats.Stats(pr_simple)
    # stats1.sort_stats(pstats.SortKey.TIME)
    # stats1.dump_stats(filename='pr_simple.prof')

    E_min, E_max = 4e-7, 4e-3
    nenergies = 5
    energy_tuple = tuple( round(n_root_scale(i, E_min, E_max, nenergies-1, n = 3), sigfigs = 11) for i in range(nenergies) )
    singlet_phase = default_singlet_phase_function(1.0)
    triplet_phase = singlet_phase + 0.2
    phases = (singlet_phase, triplet_phase,)
    so_scaling_value = 0.325
    reduced_mass = 42.47

    input_dir_name = 'RbSr+_tcpld_vs_mass'
    transfer_input_dir_name = 'RbSr+_tcpld_momentum_transfer_vs_mass'

    # output_dir = scratch_path / 'molscat' / 'outputs' / input_dir_name / f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{triplet_phase:.4f}' / f'{so_scaling_value:.4f}' / f'{reduced_mass:.4f}_amu'
    # s_matrix_collection, duration, output_dir, pickle_path = collect_and_pickle( output_dir, singlet_phase, triplet_phase, so_scaling_value, energy_tuple)

    pickle_path = pickles_dir_path / input_dir_name /f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{triplet_phase:.4f}' / f'{so_scaling_value:.4f}' / f'{reduced_mass:.4f}_amu.pickle'

    with cProfile.Profile() as pr_calc:
        time_mp_mul, time_lst_mul = measure_mp_and_lst_k_L_E(pickle_path, phases, pc = PC)

    stats2 = pstats.Stats(pr_calc)
    stats2.sort_stats(pstats.SortKey.TIME)
    stats2.dump_stats(filename='pr_calc.prof')


if __name__ == "__main__":
    main()

