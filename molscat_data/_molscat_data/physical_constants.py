# import numpy as np

hartree_in_SI                    = 4.3597447222071e-18
Boltzmann_constant_in_SI         = 1.380649e-23

# these are energy equivalents (or prefactors for the magneton values) in inverse cm [from physical_constants_module.f 2021]
hartree_in_inv_cm                = 2.1947463136320e5 # hartree_in_SI*Joule_in_inv_cm
Hz_in_inv_cm                     = 3.335640951e-11   # 1d-2/speed_of_light_in_SI
eV_in_inv_cm                     = 8.065543937e3     # electronvolt_in_SI*Joule_in_inv_cm
K_in_inv_cm                      = 0.6950348004e0    # Boltzmann_constant_in_SI*Joule_in_inv_cm
bohr_magneton_in_inv_cm_per_T    = 0.46686447783e0   # bohr_magneton_in_SI*Joule_in_inv_cm
nuclear_magneton_in_inv_cm_per_T = 2.54262341353e-4   # bohr_magneton_in_inv_cm_per_T*electron_mass/proton_mass

# Define important names :)
cm_to_Hartree = 1/hartree_in_inv_cm
MHz_to_cm = 1e6*Hz_in_inv_cm
MHz_to_Hartree = MHz_to_cm*cm_to_Hartree
K_to_cm = K_in_inv_cm
MHz_to_K = MHz_to_cm/K_to_cm
Hartree_to_K = hartree_in_inv_cm/K_in_inv_cm

# [from ]from physical_constants_module.f 2021]
atomic_mass_constant            = 1.66053906660e-27
proton_mass                     = 1.67262192369e-27
electron_mass                   = 9.1093837015e-31
amu_to_au = atomic_mass_constant/electron_mass

# vim physical_constants_module.f
bohr_in_SI                       = 0.529177210903e-10
Angstrom_in_SI                   = 1e-10
bohrtoAngstrom = bohr_in_SI/Angstrom_in_SI

bohrmagneton_MHzperG = 1.39962449361 # Bohr magneton in MHz/G

# derived
rate_from_au_to_SI = bohr_in_SI**2 * (hartree_in_SI/electron_mass)**0.5

i85Rb = 5/2
ahfs85Rb = 3035.732440300/3     # it should BE CHANGED # NOT LONGER
gs85Rb = 2.00231930     # it should be changed to fine structure Lande g-factor (in molscat input also)
gi85Rb = -2.936400e-4

i87Rb = 3/2
ahfs87Rb = 6834.68261090429/2
gi87Rb = -9.951413e-4
ge = 2.00231930

i87Sr = 9/2
ahfs87Sr = -5002.368365/5
gi87Sr = 1.09316/(i87Sr*proton_mass/electron_mass)

mass_87Rb =  86.9091805310 # NIST
mass_84Sr =  83.9134191    # NIST
mass_86Sr =  85.9092606    # NIST
mass_87Sr =  86.9088775    # NIST
mass_88Sr =  87.9056125    # NIST

red_mass_87Rb_84Sr = amu_to_au*mass_87Rb*mass_84Sr/(mass_87Rb+ mass_84Sr)
red_mass_87Rb_86Sr = amu_to_au*mass_87Rb*mass_86Sr/(mass_87Rb+ mass_86Sr)
red_mass_87Rb_87Sr = amu_to_au*mass_87Rb*mass_87Sr/(mass_87Rb+ mass_87Sr)
red_mass_87Rb_88Sr = amu_to_au*mass_87Rb*mass_88Sr/(mass_87Rb+ mass_88Sr)