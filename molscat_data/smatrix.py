from dataclasses import dataclass, field, replace
from collections import namedtuple
import warnings
from typing import ClassVar, NamedTuple
import numpy as np
import cmath
from sympy.physics.wigner import wigner_3j
# from py3nj import wigner3j as wigner_3j
# import sys
# from decimal import Decimal
from sigfig import round

import time
import timeit

import cProfile
import pstats

from physical_constants import i87Rb, ahfs87Rb, ge, gi87Rb, i87Sr, ahfs87Sr, gi87Sr, bohrmagneton_MHzperG, MHz_to_K, K_to_cm, amu_to_au, bohrtoAngstrom, Hartree_to_K
import quantum_numbers as qn


@dataclass
class SMatrix:
    """Scattering matrix for the given parameters.

    :param bool identical: if the colliding particles are identical
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
    
    identical: bool = False
    basis: tuple[str, ...] = None # ('2*L', '2*F1', '2*F2', '2*F12', '2*T', '2*MT')
    diagonal: tuple[str, ...] = None # ('MT',)

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

    def addElement(self, qn_out: tuple, qn_in: tuple, value: complex) -> None:
        """Add new elements to the S-matrix.

        :param qn_in: The quantum numbers for the initial state.
        :param qn_out: The quantum numbers for the final state.
        :param value: Value of the S-matrix element.
        :raises AssertionError: if len(qn_in) or len(qn_out) doesn't match len(basis)
        """

        assert len(qn_out) == len(qn_in) == len(self.basis), "The number of quantum numbers in both tuples should match the size of the basis."
        self.matrix[(qn_out, qn_in)] = value

    def update(self, other_S_matrix) -> None:
        """Update the S-matrix with the elements of another S-matrix if the parameters match.
        :param SMatrix S_matrix: The scattering matrix used to update the object.
        :raises AssertionError: if S_matrix has different attributes than the object or S_matrix is not an instance of the SMatrix class.
        """

        assert self == other_S_matrix, "The new S_matrix should have same attributes as the updated one."
        self.matrix.update(other_S_matrix.matrix)

    def cross_section(self, qn_out: tuple, qn_in: tuple) -> float:
        
        cross_section = np.pi/(2*self.reducedMass*self.collisionEnergy/Hartree_to_K) * np.abs((qn_out == qn_in) - self.matrix.get((qn_out, qn_in), (qn_out == qn_in)))**2 * (1+self.identical)
        return cross_section

    def rate_constant(self, qn_out: tuple, qn_in: tuple) -> float:
        rate_constant = np.sqrt(2*(self.collisionEnergy/Hartree_to_K) / self.reducedMass) * self.cross_section(qn_out, qn_in)
        return rate_constant

    def getInBasis(self, qn_out, qn_in, new_basis = None) -> complex:
        assert type(qn_out) == type(qn_in), "The types of the quantum numbers passed as arguments should be the same."
        # it would be good to replace the warning with a logger?
        if new_basis == None: new_basis = qn_in._fields; warnings.warn(f"The basis {new_basis} read from the type of qn_in argument.")
        match (self.basis, new_basis):
            case (self.basis, self.basis):
                return self.matrix[(qn_out, qn_in)]
            
            case (('L', 'F1', 'F2', 'F12', 'T', 'MT'), ('L', 'ML', 'F1', 'F2', 'F12', 'MF12')):
                """For now, it only supports matrices degenerate in MT."""
                assert 'T' in self.diagonal and 'MT' in self.diagonal, "For now, getInBasis method only supports matrices diagonal in T and MT."
                assert isinstance(qn_in, qn.LF12)
                # y = [(-1)**((qn_out.J1() + qn_in.J1() - qn_out.J23() - qn_in.J23() + qn_out.MJ1() + qn_out.MJ23() + qn_in.MJ1() + qn_in.MJ23())/2) 
                #          * np.sqrt((J123_out+1)*(J123_in+1)) 
                #          * wigner_3j(qn_out.J1()/2, qn_out.J23()/2, J123_out/2, qn_out.MJ1()/2, qn_out.MJ23()/2, -(qn_out.MJ1()/2 + qn_out.MJ23()/2)) 
                #          * wigner_3j(qn_in.J1()/2, qn_in.J23()/2, J123_in/2, qn_in.MJ1()/2, qn_in.MJ23()/2, -(qn_in.MJ1()/2 + qn_in.MJ23()/2))
                #             for J123_in in range(abs(qn_in.J1()-qn_in.J23()), qn_in.J1()+qn_in.J23()+1, 2) 
                #             for J123_out in (J123_in,)]
                x = [ (-1)**((qn_out.J1() + qn_in.J1() - qn_out.J23() - qn_in.J23() + qn_out.MJ1() + qn_out.MJ23() + qn_in.MJ1() + qn_in.MJ23())/2) 
                         * np.sqrt((J123_out+1)*(J123_in+1)) 
                         * wigner_3j(qn_out.J1()/2, qn_out.J23()/2, J123_out/2, qn_out.MJ1()/2, qn_out.MJ23()/2, -(qn_out.MJ1()/2 + qn_out.MJ23()/2)) 
                         * wigner_3j(qn_in.J1()/2, qn_in.J23()/2, J123_in/2, qn_in.MJ1()/2, qn_in.MJ23()/2, -(qn_in.MJ1()/2 + qn_in.MJ23()/2)) 
                         * self.matrix.get( ( qn.Tcpld(qn_out.J1(), qn_out.J2(), qn_out.J3(), qn_out.J23(), J123_out, 0), 
                                            qn.Tcpld(qn_in.J1(), qn_in.J2(), qn_in.J3(), qn_in.J23(), J123_in, 0) 
                                            ), 0
                                        )
                                        for J123_in in range(abs(qn_in.J1()-qn_in.J23()), qn_in.J1()+qn_in.J23()+1, 2) 
                                        for J123_out in (J123_in,)
                        ]
                # print(y, sum(y))
                # print(x)
                x = complex(sum(x))
                return x

            case (('L', 'ML', 'F1', 'F2', 'F12', 'MF12'), ('L', 'ML', 'F1', 'MF1', 'F2', 'MF2')):
                print(((qn_out.J2() + qn_in.J2() - qn_out.J3() - qn_in.J3() + qn_out.MJ2() + qn_out.MJ3() + qn_in.MJ2() + qn_in.MJ3())/2))
                x = [ (-1)**((qn_out.J2() + qn_in.J2() - qn_out.J3() - qn_in.J3() + qn_out.MJ2() + qn_out.MJ3() + qn_in.MJ2() + qn_in.MJ3())/2) 
                         * np.sqrt((J23_out+1)*(J23_in+1)) 
                         * wigner_3j(qn_out.J2()/2, qn_out.J3()/2, J23_out/2, qn_out.MJ2()/2, qn_out.MJ3()/2, -(qn_out.MJ2()/2 + qn_out.MJ3()/2)) 
                         * wigner_3j(qn_in.J2()/2, qn_in.J3()/2, J23_in/2, qn_in.MJ2()/2, qn_in.MJ3()/2, -(qn_in.MJ2()/2 + qn_in.MJ3()/2)) 
                         * self.matrix.get( ( qn.LF12(qn_out.J1(), qn_out.MJ1(), qn_out.J2(), qn_out.J3(), J23_out, qn_out.MJ2() + qn_out.MJ3()), 
                                            qn.LF12(qn_in.J1(), qn_in.MJ1(), qn_in.J2(), qn_in.J3(), J23_in, qn_in.MJ2() + qn_in.MJ3())
                                            )
                                        )
                                        for J23_in in range(abs(qn_in.J2()-qn_in.J3()), qn_in.J2()+qn_in.J3()+1, 2) 
                                        for J23_out in range(abs(qn_out.J2()-qn_out.J3()), qn_out.J2()+qn_out.J3()+1, 2)
                        ]
                # print(x)
                x = complex(sum(x))
                return x
            
            case (('L', 'F1', 'F2', 'F12', 'T', 'MT'), ('L', 'ML', 'F1', 'MF1', 'F2', 'MF2')):
                x = [ (-1)**((qn_out.J2() + qn_in.J2() - qn_out.J3() - qn_in.J3() + qn_out.MJ2() + qn_out.MJ3() + qn_in.MJ2() + qn_in.MJ3())/2) 
                         * np.sqrt((J23_out+1)*(J23_in+1)) 
                         * wigner_3j(qn_out.J2()/2, qn_out.J3()/2, J23_out/2, qn_out.MJ2()/2, qn_out.MJ3()/2, -(qn_out.MJ2()/2 + qn_out.MJ3()/2)) 
                         * wigner_3j(qn_in.J2()/2, qn_in.J3()/2, J23_in/2, qn_in.MJ2()/2, qn_in.MJ3()/2, -(qn_in.MJ2()/2 + qn_in.MJ3()/2)) 
                         * self.getInBasis( qn.LF12(qn_out.J1(), qn_out.MJ1(), qn_out.J2(), qn_out.J3(), J23_out, qn_out.MJ2() + qn_out.MJ3()), 
                                                qn.LF12(qn_in.J1(), qn_in.MJ1(), qn_in.J2(), qn_in.J3(), J23_in, qn_in.MJ2() + qn_in.MJ3()),
                                                    new_basis = ('L', 'ML', 'F1', 'F2', 'F12', 'MF12')
                                                )
                                        for J23_in in range(abs(qn_in.J2()-qn_in.J3()), qn_in.J2()+qn_in.J3()+1, 2) 
                                        for J23_out in range(abs(qn_out.J2()-qn_out.J3()), qn_out.J2()+qn_out.J3()+1, 2)
                        ]
                # print(x)
                x = complex(sum(x))
                return x

            case _:
                raise NotImplementedError(f"The transformation from {self.basis} to {new_basis} is not implemented.")


        


@dataclass
class SMatrixCollection:
    """
    Collection of the scattering matrice for the given parameters.

    :param bool identical: if the colliding particles are identical
    :param tuple basis: names of the quantum numbers in the basis.
    :param tuple C4: values of the C4 coefficients of the ion-atom potential in atomic units allowed in the collection.
    :param tuple singletParameter: values of the parameters labelling the singlet PECs allowed in the collection.
    :param tuple tripletParameter: values of the parameters labelling the triplet PECs allowed in the collection.
    :param tuple reducedMass: values of the reduced mass of the system in the atomic units allowed in the collection.
    :param tuple magneticField: values of the static magnetic field induction in gausses (G) allowed in the collection.
    :param tuple collisionEnergy: values of the energy in the center-of-mass frame of the colliding pair in kelvins (K) allowed in the collection.
    :param matrix: dict with entries of the form of (doubled_quantum_numbers_in, doubled_quantum_numbers_out): value, consisting of the S-matrix elements for the given initial and final state
    """

    identical: bool = False
    basis: tuple[str, ...] = None # doubled quantum numbers ('L', 'F1', 'F2', 'F12', 'T', 'MT')
    diagonal: tuple[str, ...] = None # ('MT')

    C4: tuple[float, ...] = None
    singletParameter: tuple[float, ...] = None
    tripletParameter: tuple[float, ...] = None
    reducedMass: tuple[float, ...] = None
    magneticField: tuple[float, ...] = None
    collisionEnergy: tuple[float, ...] = None

    # Create a namedtuple class gathering the indices of the given S-matrix within the collection
    ParametersIndices: ClassVar[ParametersIndices] = namedtuple('ParametersIndices', ('C4', 'singletParameter', 'tripletParameter', 'reducedMass', 'magneticField', 'collisionEnergy'))

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
                    self.Qn = namedtuple('Qn', basis)
                    self.diagonal = ('T', 'MT')

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
                        channels[channel_index] = self.Qn(2*int(line.split()[6]), int(line.split()[2]), int(line.split()[3]), int(line.split()[4]), T, int(line.split()[5]))
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
                    
                    S = SMatrix(basis = basis, diagonal = self.diagonal, C4 = C4, singletParameter = A_s, tripletParameter = A_t, reducedMass = reduced_mass, magneticField = magnetic_field, collisionEnergy = self.collisionEnergy[energy_counter], matrix = matrix)
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
    

def main():
    
    time_0 = time.time()
    s = SMatrixCollection.from_output(r"../data/TEST_10_ENERGIES.output")
    print(f"Loading the matrix took {time.time() - time_0} seconds.")

    
    def summing():
        Lmax = 2*19

        x = [sum([
                sum([
                        0.1*np.abs(s.matrixCollection[0,0,0,0,0,i].getInBasis(qn.LF1F2(L, ML, 2, 2, 1, -1), qn.LF1F2(L, ML, 4, 0, 1, 1)))**2 for i in range(10)
                    ])
                    for ML in range(-L, L+1, 2)
                ])
        for L in range(0, Lmax+1, 2)]

        return x
    
    time_0 = time.time()
    x = summing()
    print(f"The time was {(time.time()-time_0):.2f} s.")
    print(x)
    print(sum(x[:9]), sum(x[:19]))
    with cProfile.Profile() as pr:
        x = summing()
    
    print(x)
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    # stats.print_stats()
    stats.dump_stats(filename = '../tests/summing_stats.prof')


    # def test(qn_out, qn_in):
    #     x = [ (-1)**((qn_out.J1() + qn_in.J1() - qn_out.J23() - qn_in.J23() + qn_out.MJ1() + qn_out.MJ23() + qn_in.MJ1() + qn_in.MJ23())/2) 
    #                 * np.sqrt((J123_out+1)*(J123_in+1)) 
    #                 * wigner_3j(qn_out.J1()/2, qn_out.J23()/2, J123_out/2, qn_out.MJ1()/2, qn_out.MJ23()/2, -(qn_out.MJ1()/2 + qn_out.MJ23()/2)) 
    #                 * wigner_3j(qn_in.J1()/2, qn_in.J23()/2, J123_in/2, qn_in.MJ1()/2, qn_in.MJ23()/2, -(qn_in.MJ1()/2 + qn_in.MJ23()/2))
    #                             for J123_in in range(abs(qn_in.J1()-qn_in.J23()), qn_in.J1()+qn_in.J23()+1, 2) 
    #                             for J123_out in (J123_in,)
    #                             if abs(qn_in.MJ1() + qn_in.MJ23()) <= J123_in and abs(qn_out.MJ1() + qn_out.MJ23()) <= J123_out
    #             ]
    #     return sum(x)

    # time_0 = time.time()
    # x = [sum([
    #         sum([
    #                 0.1*np.abs(test(qn.LF12(L, ML, 2, 1, 3, -1), qn.LF12(L, ML, 4, 1, 3, 1)))**2 for i in range(10)
    #             ])
    #             for ML in range(-L, L+1, 2)
    #         ])
    #     for L in range(0, Lmax+1, 2)]
    # print(f"It took {(time.time()-time_0):.2f} s.")
    # print([float(y) for y in x])
    # # print(sum(x[:9]), sum(x[:19]), sum(x))


if __name__ == '__main__':
    main()




# time_0 = time.time()

# func = map(
#     lambda L: sum(map(
#     lambda ML: sum(map(
#     lambda i: 0.1*np.abs(s.matrixCollection[0,0,0,0,0,i].getInBasis(qn.LF1F2(L, ML, 2, 2, 1, -1), qn.LF1F2(L, ML, 4, 0, 1, 1)))**2,
#     range(10)
#     )),
#     range(-L, L+1, 2)
#     )),
#     range(0, Lmax+1, 2)
# )

# x = list(func)
# print(x)
# print(sum(x[:9]), sum(x[:19]), sum(x))
# print(f"Getting {10*((Lmax+2)/2)**2} elements in a LF1F2 basis took {time.time() - time_0} seconds.")

# time_0 = time.time()

# func = map(
#     lambda L: sum(map(
#     lambda i, ML:  0.1*np.abs(s.matrixCollection[0,0,0,0,0,i].getInBasis(qn.LF1F2(L, ML, 2, 2, 1, -1), qn.LF1F2(L, ML, 4, 0, 1, 1)))**2,
#     range(10), range(-L, L+1, 2)
#     )),
#     range(0, Lmax+1, 2)
# )

# x = list(func)
# print(x)
# print(sum(x[:9]), sum(x[:19]), sum(x))
# print(f"Getting {10*((Lmax+2)/2)**2} elements in a LF1F2 basis took {time.time() - time_0} seconds.")
# print(f"Getting {10*((Lmax+2)/2)**2} elements in a LF1F2 basis took {time.time() - time_0} seconds.")

