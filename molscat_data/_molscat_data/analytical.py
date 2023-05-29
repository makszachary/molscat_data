import numpy as np
from .physical_constants import i87Rb, ahfs87Rb, ge, gi87Rb, bohrmagneton_MHzperG, MHz_to_K, K_to_cm
from . import quantum_numbers as qn

def MonoAlkaliEnergy(magnetic_field: float, F: int, MF: int, I: int = int(2*i87Rb), a_hfs: float = ahfs87Rb, gs = ge, gi = gi87Rb):
    """Calculates the energy of the |f, mf> state of an atom in the 2S
    electronic state at given magnetic field as compared to the zero-field
    energy without hyperfine structure.
    
    :param magnetic_field: magnetic field in gausses
    :param F: doubled total angular momentum of the atom
    :param MF: doubled projection of F
    :param I: doubled nuclear spin
    :param a_hfs: hyperfine structure constant in MHz
    :param gs: electron g-factor (for the Zeeman energy = uB*(gs+gi)*B)
    :param gi: nuclear g-factor (for the Zeeman energy = uB*(gs+gi)*B)
    :return: energy of a |f, mf> state the given magnetic field in K

    """
    bohr_magneton = bohrmagneton_MHzperG

    f, mf, i = F/2, MF/2, I/2

    if not (F == I-1 or F == I + 1):
        raise ValueError("Value of  is neither I-1 nor I+1, and it has to be between |i-s| and |i+s|. Are your sure your system has spin of 1/2? This function can only treat such systems.")
    elif not ( -F <= MF <= F ):
        raise ValueError("Value of MF is less than -F or greater than F! That's impossible dude")
    elif MF == I + 1:
        energy = 1/2*a_hfs*i + gi*bohr_magneton*(i+1/2)*magnetic_field + 1/2*(gs-gi)*bohr_magneton*magnetic_field
    elif MF == -(I + 1):
        energy = 1/2*a_hfs*i - gi*bohr_magneton*(i+1/2)*magnetic_field - 1/2*(gs-gi)*bohr_magneton*magnetic_field
    elif F == I+1:
        energy = gi*bohr_magneton*mf*magnetic_field - 1/4*a_hfs + 1/2*a_hfs*(i+1/2)*np.sqrt(1 + 2*(mf*(gs-gi)*bohr_magneton*magnetic_field)/(a_hfs*(i+1/2)**2) + (((gs-gi)*bohr_magneton*magnetic_field)/(a_hfs*(i+1/2)))**2 )
    elif F == I-1:
        energy = gi*bohr_magneton*mf*magnetic_field - 1/4*a_hfs - 1/2*a_hfs*(i+1/2)*np.sqrt(1 + 2*(mf*(gs-gi)*bohr_magneton*magnetic_field)/(a_hfs*(i+1/2)**2) + (((gs-gi)*bohr_magneton*magnetic_field)/(a_hfs*(i+1/2)))**2 )

    return energy*MHz_to_K

def allThresholds(magnetic_field: float, L_min, L_max, M_tot, I1 = int(2*i87Rb), I2 = 0, identical = False):
    S1, S2 = 1, 1

    thresholds = { qn.LF1F2(L, M_tot-MF1-MF2, F1, MF1, F2, MF2): MonoAlkaliEnergy(magnetic_field, F1, MF1, I1, a_hfs = ahfs87Rb, gs = ge, gi = gi87Rb) + MonoAlkaliEnergy(magnetic_field, F2, MF2, I2, a_hfs = 0, gs = ge, gi = 0) for L in range(L_min, L_max+1, 2) for F1 in range(abs(I1 - S1), I1+S1+1, 2) for F2 in range(abs(I2-S2), I2+S2+1, 2) for MF1 in range(-F1, F1+1, 2) for MF2 in range(-F2, F2+1, 2) if (M_tot-MF1-MF2) in range(-L, L+1, 2)}

    return thresholds


def ListOfOpenChannels(B, collisionenergy, listofallchannels, f1ref, mf1ref, f2ref, mf2ref, i1 = i87Rb, i2 = 0, ahfs1 = ahfs87Rb, ahfs2 = 0, gs1 = ge, gs2 = ge, gi1 = gi87Rb, gi2 = 0, energyconversionfactor = K_to_cm):
    """
    
    Parameters:
        collisionenergy: collision energy of the colliding pair (in kelvins),
        listofchannels: list of { 'L': L, 'f1': f1, 'mf1': mf1, 'f2': f2, 'mf2': mf2, 'energy': absolute value of the channel energy (usually in cm-1)} dictionaries with increasing energies
        energyconversionfactor: name of the conversion factor allowing to convert collision energy unit to the unit of energy given in listofchannels list
    
    Results:
        list of open channels in the form of dictionaries: { 'L': L, 'f1': f1, 'mf1': mf1, 'f2': f2, 'mf2': mf2, 'energy': absolute value of the channel energy (usually in cm-1)}
    
    """
    draft = listofallchannels
    for item in draft:
        item.update({'energy': (MonoAlkaliEnergy(B, item['f1'], item['mf1'], i1, ahfs1, gs1, gi1) + MonoAlkaliEnergy(B, item['f2'], item['mf2'], i2, ahfs2, gs2, gi2))*energyconversionfactor })

    refenergy =  (MonoAlkaliEnergy(B, f1ref, mf1ref, i1, ahfs1, gs1, gi1) + MonoAlkaliEnergy(B, f2ref, mf2ref, i2, ahfs2, gs2, gi2))*energyconversionfactor
    openchannels = [item for item in draft if item['energy'] <= (refenergy + collisionenergy*energyconversionfactor)]

    return openchannels


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
