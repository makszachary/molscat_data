import numpy as np
from .physical_constants import i87Rb, ahfs87Rb, ge, gi87Rb, bohrmagneton_MHzperG, MHz_to_K

def MonoAlkaliEnergy(B, f, mf, i = i87Rb, ahfs = ahfs87Rb, gs = ge, gi = gi87Rb, bohrmagneton = bohrmagneton_MHzperG):
    """
    
    Parameters:
        magnetic field in gauss, ahfs in MHz, gi (nuclear g-factor) and gs (electron g-factor) according to a convention where the Zeeman energy is uB*(gs+gi)*B.
    
    Returns:
        hyperfine-Zeeman energy for a given magnetic field in K

    """
    #if i == 0:
    #    energy = mf*gs*bohrmagneton*B
    if not (f == i-1/2 or f == i + 1/2):
        print("Value of f is neither i-1/2 nor i+1/2, and it has to be between |i-s| and |i+s|. Are your sure your system has spin of 1/2? This function can only treat such systems.")
    elif not ( -f <= mf <= f ):
        print("Value of mf is less than -f or greater than f! That's impossible dude")
    elif mf == i + 1/2:
        energy = 1/2*ahfs*i + gi*bohrmagneton*(i+1/2)*B + 1/2*(gs-gi)*bohrmagneton*B
    elif mf == -(i + 1/2):
        energy = 1/2*ahfs*i - gi*bohrmagneton*(i+1/2)*B - 1/2*(gs-gi)*bohrmagneton*B
    elif f == i+1/2:
        energy = gi*bohrmagneton*mf*B - 1/4*ahfs + 1/2*ahfs*(i+1/2)*np.sqrt(1 + 2*(mf*(gs-gi)*bohrmagneton*B)/(ahfs*(i+1/2)**2) + (((gs-gi)*bohrmagneton*B)/(ahfs*(i+1/2)))**2 )
    elif f == i-1/2:
        energy = gi*bohrmagneton*mf*B - 1/4*ahfs - 1/2*ahfs*(i+1/2)*np.sqrt(1 + 2*(mf*(gs-gi)*bohrmagneton*B)/(ahfs*(i+1/2)**2) + (((gs-gi)*bohrmagneton*B)/(ahfs*(i+1/2)))**2 )
    else:
        print("Something went wrong!")
    try:
        return energy*MHz_to_K
    except NameError:
        print("An internal problem in MonoAlkaliEnergy function occured and the energy hasn't been calculated.")


def A_plus_squared(mf, i = 3/2, B = 2.97, B_hfs = 2441):
    return 0.5*(1+mf/(i+0.5))*(1+B/B_hfs*(1-mf/(i+0.5)))

def A_minus_squared(mf, i = 3/2, B = 2.97, B_hfs = 2441):
    return 0.5*(1-mf/(i+0.5))*(1-B/B_hfs*(1+mf/(i+0.5)))

def probabilities_from_matrix_elements_hot(f, mf, ms, i = 3/2, B = 2.97, B_hfs = 2441):
    if f != i+1/2:
        return 0
    probability = 1/4*A_plus_squared(mf = mf, i = i, B = B, B_hfs = B_hfs)*A_minus_squared(mf = mf, i = i, B = B, B_hfs = B_hfs)
    if ms == 1/2:
        probability += 1/4*A_minus_squared(mf = mf + 1, i = i, B = B, B_hfs = B_hfs)*A_minus_squared(mf = mf,  i = i, B = B, B_hfs = B_hfs)
    elif ms == -1/2:
        probability += 1/4*A_plus_squared(mf = mf - 1, i = i, B = B, B_hfs = B_hfs)*A_plus_squared(mf = mf, i = i, B = B, B_hfs = B_hfs)
    return probability

def probabilities_from_matrix_elements_cold(f, mf, ms, i = 3/2, B = 2.97, B_hfs = 2441):
    if f == i + 1/2:
        if ms == 1/2:
            return 1/4*A_plus_squared(mf = mf + 1, i = i, B = B, B_hfs = B_hfs)*A_minus_squared(mf = mf, i = i, B = B, B_hfs = B_hfs)
        elif ms == -1/2:
            return 1/4*A_minus_squared(mf = mf - 1, i = i, B = B, B_hfs = B_hfs)*A_plus_squared(mf = mf, i = i, B = B, B_hfs = B_hfs)
    elif f == i - 1/2:
        if ms == 1/2:
            return 1/4*A_minus_squared(mf = mf + 1, i = i, B = B, B_hfs = B_hfs)*A_plus_squared(mf = mf, i = i, B = B, B_hfs = B_hfs)
        elif ms == -1/2:
            return 1/4*A_plus_squared(mf = mf - 1, i = i, B = B, B_hfs = B_hfs)*A_minus_squared(mf = mf, i = i, B = B, B_hfs = B_hfs)
    return None
