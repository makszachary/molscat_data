from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

from matplotlib import pyplot as plt
import matplotlib
# matplotlib.use('TkAgg')

from _molscat_data.physical_constants import red_mass_87Rb_84Sr_amu, red_mass_87Rb_86Sr_amu, red_mass_87Rb_87Sr_amu, red_mass_87Rb_88Sr_amu
from _molscat_data.effective_probability import effective_probability, p0



pmf_path = Path(__file__).parents[1].joinpath('data', 'pmf', 'N_pdf_logic_params_EMM_500uK.txt')
pmf_array = np.loadtxt(pmf_path)


def spin_exchange_DISA(reduced_mass, max_spin_exchange, DeltaPhi88, reducedmass88Sr87Rb = red_mass_87Rb_88Sr_amu):
    spin_exchange = max_spin_exchange*np.sin(DeltaPhi88*np.sqrt(reduced_mass/red_mass_87Rb_88Sr_amu))**2
    return spin_exchange

def simple_peff_function(reduced_mass, DeltaPhi88):
    return effective_probability(spin_exchange_DISA(reduced_mass, max_spin_exchange = p0(0.45, pmf_array = pmf_array), DeltaPhi88 = DeltaPhi88), pmf_array=pmf_array)

def fit_data(f, xdata, ydata, yerr=None, bounds = None):
    popt, pcov = curve_fit(f, xdata, ydata, sigma=yerr, bounds = bounds)
    perr = np.sqrt(np.diag(pcov))
    
    # Calculate chi square (reduced)
    residuals = ydata - f(xdata, *popt)
    chisq = np.sum((residuals / yerr) ** 2) / (len(ydata) - len(popt))
    
    return popt, perr, chisq

def main():

    exp_data_path = Path(__file__).parents[1] / 'data' / 'exp_data' / 'isotopes_hpf.dat'
    exp_data = np.loadtxt(exp_data_path)[:,[0,1,3]]

    xdata = np.array([red_mass_87Rb_84Sr_amu, red_mass_87Rb_86Sr_amu, red_mass_87Rb_88Sr_amu])
    ydata = exp_data[0]
    yerr = exp_data[1]

    fitted_params, param_errors, chi_square = fit_data(simple_peff_function, xdata, ydata, yerr, [130*np.pi, 175*np.pi])
    # print(f'fitted_params = {fitted_params[0]:.4f}, {fitted_params[1]/np.pi:.4f}*pi', f'{param_errors = }', f'{chi_square = }')
    print(f'fitted_params = {fitted_params[0]/np.pi:.4f}*pi', f'{param_errors = }', f'{chi_square = }')
    print([(mu, simple_peff_function(mu, *fitted_params)) for mu in xdata])
    print([(mu, exp_data[:,i]) for i, mu in enumerate(xdata)])

    fig, ax = plt.subplots(figsize = (5, 5), dpi = 300)
    xx = np.arange(min(xdata)-0.1, max(xdata)+0.1, step = 0.01)
    ax.plot(xx, simple_peff_function(xx, *fitted_params))
    ax.scatter(xdata, ydata)
    ax.errorbar(xdata, ydata, yerr)
    fig_path = Path(__file__).parents[1] / 'plots' / 'trash' / '1.pdf'
    fig_path.parent.mkdir(parents = True, exist_ok = True)
    plt.savefig(fig_path)
    plt.close()

if __name__ == '__main__':
    main()

# Example usage:
# xdata = np.array([...])  # Your x data
# ydata = np.array([...])  # Your y data
# yerr = np.array([...])   # Your uncertainties (standard deviations)

# fitted_params, param_errors, chi_square = fit_data(fit_function, xdata, ydata, yerr)