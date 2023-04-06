import numpy as np

### Energy grids and distributions ###

def linear_scale(i, E_min, E_max, N):
    return E_min + (E_max-E_min)*(i/N)

def log_scale(i, E_min, E_max, N):
    return E_min * (E_max/E_min)**(i/N)

def sqrt_scale(i, E_min, E_max, N):
    return E_min*( (np.sqrt(E_max/E_min) - 1) / N )**2 * ( i + ( N / (np.sqrt(E_max/E_min) - 1) ) )**2

def n_root_scale(i, E_min, E_max, N, n = 3):
    return E_min*( ((E_max/E_min)**(1/n) - 1) / N )**n * ( i + ( N / ((E_max/E_min)**(1/n) - 1) ) )**n

def n_root_distribution(energy, temperature, E_min, E_max, N, n = 3):
    """
    energy in Kelvins
    temperature in Kelvins
    """
    h = (E_max**(1/n) - E_min**(1/n)) / N
    return 2*n/np.sqrt(np.pi)*(energy/temperature)**(3/2) * h *energy**(-1/n)*np.exp(-energy/temperature)

def n_root_iterator(temperature, E_min, E_max, N, n = 3):
    return (n_root_distribution(n_root_scale(i, E_min, E_max, N, n), temperature, E_min, E_max, N, n) for i in range(N))

def main():    
    x = n_root_iterator(1e-3, 1e-7, 1e-2, 100, n = 3)
    print(x)
    print(list(x))
    print([f"{n_root_scale(i, 1e-7, 1e-2, 100):.15e}" for i in range(101)])

if __name__ == '__main__':
    main()