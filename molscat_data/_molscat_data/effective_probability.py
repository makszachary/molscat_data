from pathlib import Path
from typing import Any
from collections.abc import Iterable

import numpy as np
from scipy import optimize


def effective_probability(p0: float | np.ndarray[Any, float], pmf_array: np.ndarray[(int, 2), float] = np.array([[1.0, 1.0]])) -> float | np.ndarray[Any, float]:

    p0 = np.array(p0)

    if np.any( (p0 < 0) | (p0 > 1)):
        raise ValueError(f"All values in the p0 array should be between 0 and 1. {p0=}")    

    p_eff = np.sum( [ np.sum(pmf_array[index:,1])*(1-p0)**(pmf_array[index,0]-1)*p0 for index in range(len(pmf_array)) ], axis = 0)

    return p_eff

def _effective_probability_no_error(p0: float | np.ndarray[Any, float], pmf_array: np.ndarray[(int, 2), float] = np.array([[1.0, 1.0]])) -> float | np.ndarray[Any, float]:

    p0 = np.array(p0)

    p_eff = np.sum( [ np.sum(pmf_array[index:,1])*(1-p0)**(pmf_array[index,0]-1)*p0 for index in range(len(pmf_array)) ], axis = 0)

    return p_eff

def p0(p_eff: float | np.ndarray[Any, float], pmf_array: np.ndarray[(int, 2), float] = np.array([[1.0, 1.0]])) -> float | np.ndarray[Any, float]:

    p_eff = np.array(p_eff)
    
    if np.any( (p_eff < 0) | (p_eff > 1)):
        raise ValueError("All values in the p_eff array should be between 0 and 1.")   
    
    fun = lambda x: _effective_probability_no_error(x, pmf_array = pmf_array) - p_eff
    inverse = optimize.newton(fun, np.full_like(p_eff, 0.01))
    inverse = np.around(inverse, decimals = 15)

    return inverse
    
    

def main():

    pmf_path = Path(__file__).parents[2].joinpath('data', 'pmf', 'N_pdf_logic_params_EMM_500uK.txt')
    pmf_array = np.loadtxt(pmf_path)
    prob0 = np.arange(0, 1.01, 0.1)
    print(f"p0 = {prob0} --> p_eff = {effective_probability(prob0, pmf_array)}")
    # print(len(np.array(0.2)))
    print(p0((1e-14,1e-13, 1e-12,), pmf_array = pmf_array))
    

if __name__ == '__main__':
    main()