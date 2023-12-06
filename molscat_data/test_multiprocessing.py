import multiprocessing
import time
import sys
import os
from pathlib import Path
import numpy as np

from _molscat_data.smatrix import SMatrixCollection
from _molscat_data.scaling_old import parameter_from_semiclassical_phase, default_singlet_phase_function, default_triplet_phase_function, default_singlet_parameter_from_phase, default_triplet_parameter_from_phase
from _molscat_data.utils import k_L_E_parallel, k_L_E_not_parallel

scratch_path = Path(os.path.expandvars('$SCRATCH'))

data_dir_path = Path(__file__).parents[1] / 'data'
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

def measure_mp_and_lst_mul(args: tuple[tuple[float, float], ...]):
    if sys.platform == 'win32':
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
    
    with multiprocessing.get_context('spawn').Pool(ncores) as pool:
        t0 = time.perf_counter()
        results_mp = pool.starmap(mul, args)
        time_mp = time.perf_counter()-t0
        print(f"The time of multiprocessing computations was {time_mp:.2f} s.")

    t0 = time.perf_counter()
    results_lst = [ mul(*arg) for arg in args ]
    time_lst = time.perf_counter()-t0
    print(f"The time of list computations was {time_lst:.2f} s.")
    
    return time_mp, time_lst

def measure_mp_and_lst_k_L_E(pickle_path: Path | str, phases: tuple[float, float]):
    s_matrix_collection = SMatrixCollection.fromPickle(pickle_path)

    param_indices = { "singletParameter": (s_matrix_collection.singletParameter.index(default_singlet_parameter_from_phase(phases[0])),), "tripletParameter": (s_matrix_collection.tripletParameter.index( default_triplet_parameter_from_phase(phases[1]) ), ) } if phases is not None else None
    
    dLmax = 4
    F_out, F_in, S = 2, 4, 1
    MF_out, MS_out, MF_in, MS_in = np.meshgrid(np.arange(-F_out, F_out+1, 2), np.arange(-S, S+1, 2), np.arange(-F_in, F_in+1, 2), S, indexing = 'ij')
    arg_hpf_deexcitation = (s_matrix_collection, F_out, MF_out, S, MS_out, F_in, MF_in, S, MS_in, param_indices, dLmax)

    t0 = time.perf_counter()
    _, __ = k_L_E_parallel(*arg_hpf_deexcitation)
    time_mp = time.perf_counter() - t0

    t0 = time.perf_counter()
    _, __ = k_L_E_not_parallel(*arg_hpf_deexcitation)
    time_lst = time.perf_counter() - t0

    return time_mp, time_lst

def main():

    args = tuple((a, b) for a in range(4) for b in range(4))
    # time_mp_mul, time_lst_mul = measure_mp_and_lst_mul(args)
    

    E_min, E_max = 4e-7, 4e-3
    nenergies = 5
    singlet_phase = default_singlet_phase_function(1.0)
    triplet_phase = singlet_phase + 0.2
    phases = (singlet_phase, triplet_phase,)
    so_scaling_value = 0.325
    reduced_mass = 42.47

    input_dir_name = 'RbSr+_tcpld_vs_mass'
    transfer_input_dir_name = 'RbSr+_tcpld_momentum_transfer_vs_mass'

    pickle_path = pickles_dir_path / input_dir_name /f'{E_min:.2e}_{E_max:.2e}_{nenergies}_E' / f'{singlet_phase:.4f}_{triplet_phase:.4f}' / f'{so_scaling_value:.4f}' / f'{reduced_mass:.4f}_amu.pickle'

    time_mp_mul, time_lst_mul = measure_mp_and_lst_k_L_E(pickle_path, phases)


if __name__ == "__main__":
    main()

