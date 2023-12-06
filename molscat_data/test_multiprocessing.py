import multiprocessing
import time
import sys
import os


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

def main():

    args = tuple((a, b) for a in range(4) for b in range(4))
    time_mp, time_lst = measure_mp_and_lst_mul(args)
    


if __name__ == "__main__":
    main()

