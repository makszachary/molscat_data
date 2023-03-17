from dataclasses import dataclass, field, replace
from collections import namedtuple
import numpy as np
import cmath
# import sys
# from decimal import Decimal
from sigfig import round

import time
import timeit

from physical_constants import i87Rb, ahfs87Rb, ge, gi87Rb, i87Sr, ahfs87Sr, gi87Sr, bohrmagneton_MHzperG, MHz_to_K, K_to_cm, amu_to_au, bohrtoAngstrom, Hartree_to_K



@dataclass
class SMatrix:
    """Scattering matrix for the given parameters.

    :param tuple basis: names of the quantum numbers in the basis.
    :param float C4: C4 coefficient of the ion-atom potential in atomic units.
    :param float singletParameter: parameter labelling the singlet PECs.
    :param float tripletParameter: parameter labelling the triplet PECs.
    :param float reducedMass: reduced mass of the system in the atomic units.
    :param float magneticField: static magnetic field induction in gausses (G).
    :param float collisionEnergy: energy in the center-of-mass frame of the colliding pair in kelvins (K).
    :param dict matrix: dict with entries of the form of (quantum_numbers_in, quantum_numbers_out): value, consisting of the S-matrix elements for the given initial and final state.
    :raises AssertionError: if len(key[0]) or len(key[1]) for any key in the matrix doesn't match len(basis).
    """

    basis: tuple[str, ...] = None # ('2*L', '2*F1', '2*F2', '2*F12', '2*T', '2*MT')

    C4: float = None
    singletParameter: float = None
    tripletParameter: float = None
    reducedMass: float = None
    magneticField: float = None
    collisionEnergy: float = None

    matrix: dict[tuple[tuple[int, ...], tuple[int, ...]], complex] = field(default_factory = dict, compare = False, repr = False)


    def __post_init__(self):
        """Check if the number of quantum numbers in the matrix match the length of the basis."""
        assert all(len(key[0]) == len(key[1]) == len(self.basis) for key in self.matrix.keys()), "The number of quantum numbers in both tuples should match the size of the basis."

    def __add__(self, other_S_matrix):
        """Merge two S-matrices of the same attributes.

        :param SMatrix other_S_matrix: the S-matrix to be added.
        :raises AssertionError: if the S-matrices have different attributes.
        :return: new SMatrix object.
        """

        assert self == other_S_matrix, "The new S_matrix should have same attributes as the updated one."

        attributes = vars(self).copy()
        attributes.pop('matrix')
        matrix = self.matrix.copy()
        matrix.update(other_S_matrix.matrix)
        
        return SMatrix(**attributes, matrix = matrix)

    def addElement(self, qn_out: tuple, qn_in: tuple, value: complex):
        """Add new elements to the S-matrix.

        :param qn_in: The quantum numbers for the initial state.
        :param qn_out: The quantum numbers for the final state.
        :param value: Value of the S-matrix element.
        :raises AssertionError: if len(qn_in) or len(qn_out) doesn't match len(basis)
        """

        assert len(qn_out) == len(qn_in) == len(self.basis), "The number of quantum numbers in both tuples should match the size of the basis."
        self.matrix[(qn_out, qn_in)] = value

    def update(self, other_S_matrix):
        """Update the S-matrix with the elements of another S-matrix if the parameters match.
        :param SMatrix S_matrix: The scattering matrix used to update the object.
        :raises AssertionError: if S_matrix has different attributes than the object or S_matrix is not an instance of the SMatrix class.
        """

        assert self == other_S_matrix, "The new S_matrix should have same attributes as the updated one."
        self.matrix.update(other_S_matrix.matrix)


@dataclass
class SMatrixCollection:
    """
    Collection of the scattering matrice for the given parameters.

    :param tuple basis: names of the quantum numbers in the basis.
    :param tuple C4: values of the C4 coefficients of the ion-atom potential in atomic units allowed in the collection.
    :param tuple singletParameter: values of the parameters labelling the singlet PECs allowed in the collection.
    :param tuple tripletParameter: values of the parameters labelling the triplet PECs allowed in the collection.
    :param tuple reducedMass: values of the reduced mass of the system in the atomic units allowed in the collection.
    :param tuple magneticField: values of the static magnetic field induction in gausses (G) allowed in the collection.
    :param tuple collisionEnergy: values of the energy in the center-of-mass frame of the colliding pair in kelvins (K) allowed in the collection.
    :param matrix: dict with entries of the form of (doubled_quantum_numbers_in, doubled_quantum_numbers_out): value, consisting of the S-matrix elements for the given initial and final state
    """

    basis: tuple[str, ...] = None # doubled quantum numbers ('L', 'F1', 'F2', 'F12', 'T', 'MT')

    C4: tuple[float, ...] = None
    singletParameter: tuple[float, ...] = None
    tripletParameter: tuple[float, ...] = None
    reducedMass: tuple[float, ...] = None
    magneticField: tuple[float, ...] = None
    collisionEnergy: tuple[float, ...] = None

    # Create a namedtuple class gathering the indices of the given S-matrix within the collection
    ParametersIndices = namedtuple('ParametersIndices', ('C4', 'singletParameter', 'tripletParameter', 'reducedMass', 'magneticField', 'collisionEnergy'))

    matrixCollection: dict[ParametersIndices, SMatrix] = field(default_factory = dict, compare = False)

    def __add__(self, s_collection):
        """Merge two S-matrix collections of the same attributes.

        :param SMatrixCollection s_collection: the S-matrix collection to be added.
        :raises AssertionError: if the S-matrix collections have different attributes.
        :return: new SMatrixCollection object.
        """

        assert self == s_collection, "The new S_matrix_collection should have same attributes as the updated one."

        attributes = vars(self).copy()
        attributes.pop('matrixCollection')
        collection = self.matrixCollection.copy()
        collection.update(s_collection.matrixCollection)

        return SMatrixCollection(**attributes, matrixCollection = collection)

    def update(self, s_collection):
        """Update the S-matrix collection with the S-matrices from another S-matrix collection if the attributes match.
        :param SMatrixCollection s_collection: The S-matrix collection used to update the object.
        :raises AssertionError: if s_colection has different attributes than the object or s_collection is not an instance of the SMatrixCollection class.
        """

        assert self == s_collection, "The new S_matrix should have same attributes as the updated one."
        self.matrixCollection.update(s_collection.matrixCollection)

    def update_from_output(self, file_path: str):
        """Update the S-matrix collection with data from a single molscat output file in tcpld basis.

        :param file_path: Path to the molscat output file.
        """
        
        with open(file_path,'r') as molscat_output:
            for line in molscat_output:
                if "REDUCED MASS FOR INTERACTION =" in line:
                    reduced_mass = float(line.split()[5])*amu_to_au # reduced mass is given in amu in MOLSCAT output/input, we convert it to atomic units
                    if self.reducedMass == None: self.reducedMass = (reduced_mass,)
                    assert reduced_mass in self.reducedMass, f"The reduced mass in the molscat output should be an element of {self}.reducedMass."
                    reduced_mass_index = self.reducedMass.index(reduced_mass)

                # determine which basis set is used in the output
                elif "in a total angular momentum basis" in line:
                    tcpld = True
                    basis = ('L', 'F1', 'F2', 'F12', 'T', 'MT')
                    Qn = namedtuple('Qn', basis)

                    if self.basis == None: self.basis = basis
                    assert basis == self.basis, f"The basis set used in the molscat output should match {self}.basis."

                elif "INTERACTION TYPE IS  ATOM - ATOM IN UNCOUPLED BASIS" in line:
                    raise NotImplementedError("Only the (L F1 F2 F12 T MT) basis can be used in the molscat outputs in the current implementation.")
                    tcpld = False
                    basis = ('L', 'ML', 'F1', 'MF1', 'F2', 'MF2')

                    if self.basis == None: self.basis = basis
                    assert basis == self.basis, f"The basis set used in the molscat output should match {self}.basis."
                
                elif "SHORT-RANGE POTENTIAL 1 SCALING FACTOR" in line:
                    #  find values of short-range factors and C4
                    A_s = round(float(line.split()[9])*float(line.split()[11]), sigfigs = 12)      
                    if self.singletParameter == None: self.singletParameter = (A_s,)
                    assert A_s in self.singletParameter, f"The singlet scaling parameter from the molscat output should be an element of {self}.singletParameter."
                    A_s_index = self.singletParameter.index(A_s)

                    for i in range(2):
                        line = next(molscat_output)
                        if "C 4 =" in line:
                            C4 = float(line.split()[3])
                    if self.C4 == None: self.C4 = (C4,)
                    assert C4 in self.C4, f"The value of C4 from the molscat output should be an element of {self}.C4."
                    C4_index = self.C4.index(C4)
                
                elif "SHORT-RANGE POTENTIAL 2 SCALING FACTOR" in line:
                    A_t = round(float(line.split()[9])*float(line.split()[11]), sigfigs = 12)
                    
                    if self.tripletParameter == None: self.tripletParameter = (A_t,)
                    assert A_t in self.tripletParameter, f"The triplet scaling parameter from the molscat output should be an element of {self}.tripletParameter."
                    A_t_index = self.tripletParameter.index(A_t)

                elif "INPUT ENERGY LIST IS" in line:
                    while "CALCULATIONS WILL BE PERFORMED FOR" not in line: line = next(molscat_output)
                    line = next(molscat_output)
                    # create the list of energies from the output
                    energy_list = []
                    # append each energy value from the output to the list of energies
                    while line.strip():
                        energy_list.append(float(line.split()[6]))
                        line = next(molscat_output)
                    energy_tuple = tuple(energy_list)

                    if self.collisionEnergy == None: self.collisionEnergy = energy_tuple
                    assert energy_tuple == self.collisionEnergy, f"The list of collision energies from the molscat output should be an element of {self}.energy_tuple."

                # elif "THESE ENERGY VALUES ARE RELATIVE TO THE REFERENCE ENERGY SPECIFIED BY MONOMER QUANTUM NUMBERS" in line:
                #     f1ref, mf1ref, f2ref, mf2ref = int(line.split()[14])/2, int(line.split()[15])/2, int(line.split()[16])/2, int(line.split()[17])/2 

                elif "*****************************  ANGULAR MOMENTUM JTOT  =" in line and tcpld:
                    T = int(line.split()[5])  
                    # print(T)              
                    energy_counter = 0

                elif "MAGNETIC Z FIELD =" in line:
                    magnetic_field = float(line.split()[13])

                    if self.magneticField == None: self.magneticField = (magnetic_field,)
                    assert magnetic_field in self.magneticField, f"The magnetic field from the molscat output should be an element of {self}.magneticField."
                    magnetic_field_index = self.magneticField.index(magnetic_field)

                # create the list of channels from the output of molscat-RKHS-tcpld
                elif "BASIS FUNCTION LIST:" in line and tcpld:
                    channels = {}
                    line = next(molscat_output)
                    line = next(molscat_output)
                    line = next(molscat_output)
                    while line.strip():
                        channel_index = int(line.split()[0])
                        channels[channel_index] = Qn(2*int(line.split()[6]), int(line.split()[2]), int(line.split()[3]), int(line.split()[4]), T, int(line.split()[5]))
                        line = next(molscat_output)
                
                elif "OPEN CHANNEL   WVEC (1/ANG.)    CHANNEL" in line and tcpld:
                    line = next(molscat_output)
                    # create a list of indices of the channels which match the declared collision energy
                    channel_in_indices = []
                    while line.strip():
                        # convert the wavevector from 1/angstrom to 1/bohr
                        wavevector = float(line.split()[1])*bohrtoAngstrom
                        # convert the channel collision energy from hartrees to kelvins
                        channel_collision_energy = Hartree_to_K*(wavevector)**2/(2*reduced_mass)
                        # get the index of the channel
                        channel_index = int(line.split()[2])
                        # append the index of the channel if matching the collision energy with the tolerance of 1e-6
                        if np.around(channel_collision_energy/self.collisionEnergy[energy_counter] - 1, decimals = 6) == 0: channel_in_indices.append(channel_index)
                        line = next(molscat_output)
                
                elif "ROW  COL       S**2                  PHASE/2PI" in line:
                    line = next(molscat_output)
                    matrix = {}
                    
                    while line.strip():
                        channel_out_index, channel_in_index = int(line.split()[0]), int(line.split()[1])
                        if channel_in_index in channel_in_indices:
                            matrix[(channels[channel_out_index], channels[channel_in_index])] = cmath.rect(np.sqrt(float(line.split()[2])), 2*np.pi*float(line.split()[3]))
                        line = next(molscat_output)
                    
                    S = SMatrix(reducedMass = reduced_mass, magneticField = magnetic_field, C4 = C4, singletParameter = A_s, tripletParameter = A_t, collisionEnergy = self.collisionEnergy[energy_counter], basis = basis, matrix = matrix)
                    if (C4_index, A_s_index, A_t_index, reduced_mass_index, magnetic_field_index, energy_counter) in self.matrixCollection.keys():
                        self.matrixCollection[self.ParametersIndices(C4_index, A_s_index, A_t_index, reduced_mass_index, magnetic_field_index, energy_counter)].update(S)
                    else:
                        self.matrixCollection[self.ParametersIndices(C4_index, A_s_index, A_t_index, reduced_mass_index, magnetic_field_index, energy_counter)] = S
                    energy_counter +=1

    @classmethod
    def from_output(cls, file_path: str):
        s_collection = SMatrixCollection()
        s_collection.update_from_output(file_path = file_path)
        return s_collection



# s_coll_1 = SMatrixCollection.from_output(r"data/TEST.output")
# print(s_coll_1)
# s_coll_1.update_from_output(r"data/TEST_3_ENERGIES.output")
# 
# s_coll_2 = SMatrixCollection.from_output(r"data/TEST_3_ENERGIES.output")
# print(s_coll_2)
time_0 = time.time()
s = SMatrixCollection.from_output(r"../data/TEST_10_ENERGIES.output")
print(s)
# print(s.matrixCollection[(0,0,0,0,0,0)].matrix)
print(f"Loading the matrix took {time.time() - time_0} seconds.")

print(timeit.timeit(lambda: SMatrixCollection.from_output(r"../data/TEST_10_ENERGIES.output"), setup = "from __main__ import SMatrix, SMatrixCollection", number = 10)/10)

# s1 = SMatrix(basis = ('L','mf1', 'mf2'), matrix={((2,3,4),(3,5,4)): 0.11})
# s2 = SMatrix(basis = ('L','mf1', 'mf2'), matrix={((2,3,6),(3,5,4)): 0.008+0.2j})
# print(s1)
# # print(s1.matrix)
# # print(s2.matrix)
# # # s1.addElement((2,3,11,2,3,4), (2,3,11,2,1,6), 0.98*np.exp(3/5*np.pi*1j))
# # s1.addElement((2,3,4), (2,3,11), 0.98*np.exp(3/5*np.pi*1j))
# # print(s1)
# # print(s1.__dict__)

# s3 = s1 + s2
# print(s1)
# print(s2)
# print(s3)
# s1.update(s2)
# print(s1)

# s2 = SMatrixCollection().load_from_output()
# print(s2)