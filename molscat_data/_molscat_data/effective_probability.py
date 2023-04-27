from pathlib import Path
from typing import Any

import numpy as np
from scipy import optimize


def effective_probability(p0: float | np.ndarray[Any, float], pmf_array: np.ndarray[(int, 2), float] = np.array([[1.0, 1.0]])) -> float | np.ndarray[Any, float]:

    p0 = np.array(p0)

    if np.any( (p0 < 0) | (p0 > 1)):
        raise ValueError("All values in the p0 array should be between 0 and 1.")    

    # p_eff = sum([ sum(pmf_array[index:,1])*(1-p0)**(pmf_array[index,0]-1)*p0 for index in range(len(pmf_array)) ])
    p_eff = np.sum( [ np.sum(pmf_array[index:,1])*(1-p0)**(pmf_array[index,0]-1)*p0 for index in range(len(pmf_array)) ], axis = 0)

    return p_eff

def p0(peff: float | np.ndarray[Any, float], pmf_array: np.ndarray[(int, 2), float] = np.array([[1.0, 1.0]])) -> float | np.ndarray[Any, float]:
    fun = lambda x: effective_probability(x, pmf_array = pmf_array) - peff
    inverse = optimize.newton(fun, 0.01)
    return inverse
    

def main():

    pmf_path = Path(__file__).parents[2].joinpath('data', 'pmf', 'N_pdf_logic_params_EMM_500uK.txt')
    pmf_array = np.loadtxt(pmf_path)
    p0 = np.arange(0, 1.01, 0.1)
    print(f"p0 = {p0} --> p_eff = {effective_probability(p0, pmf_array)}")
    

if __name__ == '__main__':
    main()