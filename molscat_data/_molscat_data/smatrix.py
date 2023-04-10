from dataclasses import dataclass, field, replace
import warnings
from typing import NamedTuple, Any, Iterable
import numpy as np
# import numpy.typing as npt
import cmath
import scipy
from py3nj import clebsch_gordan
from sigfig import round
import json
import msgpack
import pickle

import itertools

import time
# import timeit

import cProfile
import pstats

from .physical_constants import i87Rb, ahfs87Rb, ge, gi87Rb, i87Sr, ahfs87Sr, gi87Sr, bohrmagneton_MHzperG, MHz_to_K, K_to_cm, amu_to_au, bohrtoAngstrom, Hartree_to_K, rate_from_au_to_SI
from . import quantum_numbers as qn
from .thermal_averaging import n_root_iterator


class CollectionParameters(NamedTuple):
    C4: float | tuple[float, ...]
    singletParameter: float | tuple[float, ...]
    tripletParameter: float | tuple[float, ...]
    reducedMass: float | tuple[float, ...]
    magneticField: float | tuple[float, ...]
    collisionEnergy: float | tuple[float, ...]


class CollectionParametersIndices(NamedTuple):
    C4: int | tuple[int, ...]
    singletParameter: int | tuple[int, ...]
    tripletParameter: int | tuple[int, ...]
    reducedMass: int | tuple[int, ...]
    magneticField: int | tuple[int, ...]
    collisionEnergy: int | tuple[int, ...]


class SMatrixCollectionEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, complex):
            return { '__complex__': True, 'abs': abs(o), 'phase': cmath.phase(o) }
        elif isinstance(o, SMatrix) or isinstance(o, SMatrixCollection):
            return o.encodableForm()
        return super().default(self, o)
 

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


    # def __post_init__(self) -> None:
    #     """Check if the number of quantum numbers in the matrix match the length of the basis."""
    #     assert all(len(key[0]) == len(key[1]) == len(self.basis) for key in self.matrix.keys()), "The number of quantum numbers in both tuples should match the size of the basis."

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

        :param qn_out: The quantum numbers for the final state.
        :param qn_in: The quantum numbers for the initial state.
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


    def encodableMatrix(self) -> dict:
        encodable_matrix = { "__SMatrix__": True }
        encodable_matrix.update({ str(tuple(tuple(qn) for qn in key)): value for key, value in self.matrix.items() })
        return encodable_matrix


    def encodableForm(self) -> dict:
        attributes = vars(self).copy()
        attributes.pop('matrix')
        
        for attr, value in attributes.items():
            if isinstance(value, tuple):
                attributes[attr] = str(value)

        s_matrix = self.encodableMatrix()
        s_matrix.pop('__SMatrix__')

        encodable_form = { "__SMatrix__": True }
        encodable_form.update( self.__class__(**attributes, matrix = s_matrix ).__dict__ )
        
        return encodable_form

    @staticmethod
    def default(o):
        if isinstance(o, complex):
            return { '__complex__': True, 'abs': abs(o), 'phase': cmath.phase(o) }
        elif isinstance(o, SMatrix) or isinstance(o, SMatrixCollection):
            return o.encodableForm()
        return o

    def toJSON(self, file_path: str) -> None:
        with open(file_path, 'w') as file:
            json.dump(self, file, default = self.default, indent = 3)

    @classmethod
    def decode(cls, dct):
        if '__complex__' in dct:
            return cmath.rect(dct['abs'], dct['phase'])
        
        elif '__SMatrix__' in dct:
            matrix = dct.get('matrix')
            
            ## parsing basis name
            for key, value in dct.items():
                if isinstance(value, str) and value.startswith('(') and value.endswith(')'):
                    dct[key] = eval(value)

            if dct['basis'] == ('L', 'F1', 'F2', 'F12', 'T', 'MT'):
                new_matrix = { ( qn.Tcpld( *eval(key)[0] ), qn.Tcpld( *eval(key)[1] ) ): value for key, value in matrix.items() }
            
            dct.pop('matrix')
            dct.pop('__SMatrix__')

            return cls(**dct, matrix = new_matrix)
        
        return dct


    @classmethod
    def fromJSON(cls, file_path):
        with open(file_path, 'r') as file:
            s_matrix = json.load(file, object_hook = cls.decode)
            return s_matrix


    def getInBasis(self, qn_out: tuple | qn.LF1F2 | qn.LF12 | qn.Tcpld, qn_in: tuple | qn.LF1F2 | qn.LF12 | qn.Tcpld, basis: str = None) -> complex:
        """Get the S-matrix element for in the given basis.

        :param qn_out: The quantum numbers for the final state.
        :param qn_in: The quantum numbers for the initial state.
        :param basis: Possible values: ('L', 'ML', 'F1', 'MF1', 'F2', 'MF2'),
        ('L', 'ML', 'F1', 'F2', 'F12', 'MF12'), ('L', 'F1', 'F2', 'F12', 'T', 'MT').
        If None (default), then inferred from the data type of qn_in argument.
        :raises TypeError: if the data types of qn_out and qn_in don't match.
        :raises AssertionError: read the source code.
        """

        if type(qn_out) != type(qn_in):
            raise TypeError(f"The types of the quantum numbers passed as arguments should be the same. You passed {qn_out =}: {type(qn_out)} and {qn_in =}: {type(qn_in)}.")
        # it would be good to replace the warning with a logger?
        if basis == None: basis = qn_in._fields
        match (self.basis, basis):
            case (self.basis, self.basis):
                return self.matrix.get((qn_out, qn_in), 0)
            
            case (('L', 'F1', 'F2', 'F12', 'T', 'MT'), ('L', 'ML', 'F1', 'F2', 'F12', 'MF12')):
                """For now, it only supports matrices degenerate in MT."""
                assert 'T' in self.diagonal and 'MT' in self.diagonal, "For now, getInBasis method only supports matrices diagonal in T and MT."
                assert isinstance(qn_in, qn.LF12)
                # we have to exclude the total-momentum-changing collisions manually
                # because for all the other in/out channels we assume that S(MT, MT') = S(0, 0)
                if qn_out.MJ1() + qn_out.MJ23() != qn_in.MJ1() + qn_in.MJ23():
                    return 0
                # we force the MJ123 to be equal to MJ1+MJ23, but abs(MJ123) has to be less than J123
                # so we limit the values of J123 to those larger or equal to abs(MJ123)
                # if the matrix is diagonal in J123, then J123_out = J123_in
                # so we have to limit both J123_out and J123_in to those l.o.e. abs(MJ123_out) and abs(MJ123_in)
                J123_in = np.arange(max(abs(qn_in.J1()-qn_in.J23()), abs(qn_in.MJ1()+qn_in.MJ23()), abs(qn_out.MJ1()+qn_out.MJ23())), qn_in.J1()+qn_in.J23()+1, 2, dtype = np.int16) # max 2**15 values
                J123_out = J123_in

                J_in = np.full((J123_in.shape[0], 6), [qn_in.J1(), qn_in.J23(), 0, qn_in.MJ1(), qn_in.MJ23(), qn_in.MJ1()+qn_in.MJ23()]).transpose()
                J_in[2] = J123_in
                J_out = np.full((J123_out.shape[0], 6), [qn_out.J1(), qn_out.J23(), 0, qn_out.MJ1(), qn_out.MJ23(), qn_out.MJ1()+qn_out.MJ23()]).transpose()
                J_out[2] = J123_out

                matrix = np.array([ self.matrix.get( (qn.Tcpld(qn_out.J1(), qn_out.J2(), qn_out.J3(), qn_out.J23(), J_out[2][k], 0), 
                                                qn.Tcpld(qn_in.J1(), qn_in.J2(), qn_in.J3(), qn_in.J23(), J_in[2][k], 0) 
                                                    ), 0
                                                ) for k in range(len(J_out[2]))])

                x = clebsch_gordan(*J_out)*clebsch_gordan(*J_in)*matrix

                x = np.sum(x)
                return x
            
            case (('L', 'F1', 'F2', 'F12', 'T', 'MT'), ('L', 'ML', 'F1', 'MF1', 'F2', 'MF2')) | (('L', 'ML', 'F1', 'F2', 'F12', 'MF12'), ('L', 'ML', 'F1', 'MF1', 'F2', 'MF2')):
                # we force the MJ123 to be equal to MJ1+MJ23, but abs(MJ123) has to be less than J123
                # so we limit the values of J123 to those larger or equal to abs(MJ123)
                # if the matrix is diagonal in J123, then J123_out = J123_in
                # so we have to limit both J123_out and J123_in to those l.o.e. abs(MJ123_out) and abs(MJ123_in)
                J23_in = np.arange(max(abs(qn_in.J2()-qn_in.J3()), abs(qn_in.MJ2()+qn_in.MJ3())), qn_in.J2()+qn_in.J3()+1, 2, dtype = np.int16) # max 2**15
                J23_out = np.arange(max(abs(qn_out.J2()-qn_out.J3()), abs(qn_out.MJ2()+qn_out.MJ3())), qn_out.J2()+qn_out.J3()+1, 2, dtype = np.int16) # max 2**15
                J23_out, J23_in = np.meshgrid(J23_out, J23_in)

                J_in = np.full((*J23_in.shape, 6), [qn_in.J2(), qn_in.J3(), 0, qn_in.MJ2(), qn_in.MJ3(), qn_in.MJ2()+qn_in.MJ3()]).transpose(2,0,1)
                J_in[2] = J23_in
                J_out = np.full((*J23_out.shape, 6), [qn_out.J2(), qn_out.J3(), 0, qn_out.MJ2(), qn_out.MJ3(), qn_out.MJ2()+qn_out.MJ3()]).transpose(2,0,1)
                J_out[2] = J23_out


                matrix = np.array([ [self.getInBasis( qn.LF12(qn_out.J1(), qn_out.MJ1(), qn_out.J2(), qn_out.J3(), J_out[2][k][j], J_out[5][k][j]), 
                                            qn.LF12(qn_in.J1(), qn_in.MJ1(), qn_in.J2(), qn_in.J3(), J_in[2][k][j], J_in[5][k][j]),
                                                basis = ('L', 'ML', 'F1', 'F2', 'F12', 'MF12') 
                                                )
                                    for j in range(J_in.shape[2])]
                                    for k in range(J_in.shape[1])
                                    ])

                x = clebsch_gordan(*J_out)*clebsch_gordan(*J_in)*matrix
                x = np.sum(x)
                return x

            case _:
                raise NotImplementedError(f"The transformation from {self.basis} to {basis} is not implemented.")


    def getCrossSection(self, qn_out: tuple | qn.LF1F2 | qn.LF12 | qn.Tcpld, qn_in: tuple | qn.LF1F2 | qn.LF12 | qn.Tcpld, basis: str = None) -> float:
        """Calculate the cross section in the given basis.

        :param qn_out: quantum numbers for the final state.
        :param qn_in: quantum numbers for the initial state.
        :param basis: Possible values: ('L', 'ML', 'F1', 'MF1', 'F2', 'MF2'),
        ('L', 'ML', 'F1', 'F2', 'F12', 'MF12'), ('L', 'F1', 'F2', 'F12', 'T', 'MT').
        If None (default), then inferred from the data type of qn_in argument.
        """
       
        arguments = locals().copy(); del arguments['self']
        S_ij = self.getInBasis(**arguments)
        if S_ij == 0:
            return 0

        cross_section = np.pi/(2*self.reducedMass*self.collisionEnergy/Hartree_to_K) * np.abs((qn_out == qn_in) - S_ij)**2 * (1+self.identical)

        return cross_section


    def getMomentumTransferCrossSection(self, qn_in: tuple | qn.LF1F2 | qn.LF12 | qn.Tcpld, basis: str = None) -> float:
        """Calculate the momentum-transfer cross section in the given basis.
        
        :param qn_in: quantum numbers for the initial state.
        Ignores all the values of L, ML passed in qn_in.
        :param basis: Possible values: ('L', 'ML', 'F1', 'MF1', 'F2', 'MF2'),
        ('L', 'ML', 'F1', 'F2', 'F12', 'MF12').
        If None (default), then inferred from the data type of qn_in argument.

        Assumes the collision is fully elastic.
        Uses the expression from [Phys. Rev. A 89, 052705 (2014)](https://doi.org/10.1103/PhysRevA.89.052705).
        """


        L_max = max( [ qns[1].L for qns in self.matrix.keys() ])
        if basis == None: basis = qn_in._fields
        match basis:
            case ('L', 'ML', 'F1', 'F2', 'F12', 'MF12') | ('L', 'ML', 'F1', 'MF1', 'F2', 'MF2'):
                qn_ins = (qn_in.__class__(L, 0, *qn_in[2:]) for L in range(0, L_max+1, 2))
            case ('L', 'F1', 'F2', 'F12', 'T', 'MT'):
                raise NotImplementedError(f"The totally coupled basis {basis} cannot be used - it would require decoupling L and F12 before using the expression for the momentum-transfer cross section.")
            case _:
                raise NotImplementedError(f"Case {basis=} is not implemented.")

        S = np.array( [ self.getInBasis(qn_out = qn, qn_in = qn, basis = basis) for qn in qn_ins ] )
        phase_shift = np.angle(S)/2

        cross_section = np.fromiter((2*L+2 for L in range(0, L_max, 2)), float) * np.sin(phase_shift[:-1])**2 - np.fromiter((2*L+4 for L in range(0, L_max, 2)), float) * np.sin(phase_shift[:-1]) * np.sin(phase_shift[1:]) * np.cos(phase_shift[:-1] - phase_shift[1:])
        cross_section *= 2*np.pi/(2*self.reducedMass*(self.collisionEnergy/Hartree_to_K))
        cross_section *= (1+self.identical)
        cross_section = cross_section.sum() 
        
        return cross_section


    def getRateCoefficient(self, qn_out: tuple | qn.LF1F2 | qn.LF12 | qn.Tcpld, qn_in: tuple | qn.LF1F2 | qn.LF12 | qn.Tcpld, basis: str = None, unit : str = None) -> float:
        """Calculate the rate coefficient in the given basis.

        :param qn_out: quantum numbers for the final state.
        :param qn_in: quantum numbers for the initial state.
        :param basis: Possible values: ('L', 'ML', 'F1', 'MF1', 'F2', 'MF2'),
        ('L', 'ML', 'F1', 'F2', 'F12', 'MF12'), ('L', 'F1', 'F2', 'F12', 'T', 'MT').
        If None (default), then inferred from the data type of qn_in argument.
        :param unit: Name of the unit at the output if other than a.u.
        Possible values: 'cm**3/s', 'a.u.', None (default).

        """

        arguments = locals().copy(); del arguments['self']; del arguments['unit']

        rate_coefficient = np.sqrt(2*(self.collisionEnergy/Hartree_to_K)/self.reducedMass) * self.getCrossSection(**arguments)
        
        match unit:
            case 'cm**3/s':
                rate_coefficient *= 10**6 * rate_from_au_to_SI
            case None | 'a.u.':
                pass
            case _:
                raise ValueError(f"The possible values of unit are: 'cm**3/s', 'a.u.', None (default), {unit=} matching none of these.")

        return rate_coefficient


    def getMomentumTransferRateCoefficient(self, qn_in: tuple | qn.LF1F2 | qn.LF12 | qn.Tcpld, basis: str = None, unit : str = None) -> float:
        """Calculate the momentum transfer rate coefficient in the given basis.

        :param qn_in: quantum numbers for the initial state.
        Ignores all the values of L, ML passed in qn_in.
        :param basis: Possible values: ('L', 'ML', 'F1', 'MF1', 'F2', 'MF2'),
        ('L', 'ML', 'F1', 'F2', 'F12', 'MF12'), ('L', 'F1', 'F2', 'F12', 'T', 'MT').
        If None (default), then inferred from the data type of qn_in argument.
        :param unit: Name of the unit at the output if other than a.u.
        Possible values: 'cm**3/s', 'a.u.', None (default).
        
        Assumes the collision is fully elastic.
        Uses the expression from [Phys. Rev. A 89, 052705 (2014)](https://doi.org/10.1103/PhysRevA.89.052705).

        """

        arguments = locals().copy(); del arguments['self']; del arguments['unit']

        rate_coefficient = np.sqrt(2*(self.collisionEnergy/Hartree_to_K)/self.reducedMass) * self.getMomentumTransferCrossSection(**arguments)
        
        match unit:
            case 'cm**3/s':
                rate_coefficient *= 10**6 * rate_from_au_to_SI
            case None | 'a.u.':
                pass
            case _:
                raise ValueError(f"The possible values of unit are: 'cm**3/s', 'a.u.', None (default), {unit=} matching none of these.")

        return rate_coefficient

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
    :param matrixCollection: dict with entries of the form of CollectionParametersIndices(C4_index, ..., collisionEnergy_index): SMatrix
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
    # ParametersIndices: ClassVar[ParametersIndices] = namedtuple('ParametersIndices', ('C4', 'singletParameter', 'tripletParameter', 'reducedMass', 'magneticField', 'collisionEnergy'))

    matrixCollection: dict[CollectionParametersIndices, SMatrix] = field(default_factory = dict, compare = False)

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
                    # reduced mass is given F14.9 format in amu in MOLSCAT output, we convert it to atomic units and round
                    reduced_mass = round(float(line.split()[5])*amu_to_au, sigfigs = 8)
                    if self.reducedMass == None: self.reducedMass = (reduced_mass,)
                    rounded_reducedMass = tuple(round(reduced_mass, sigfigs = 8) for reduced_mass in self.reducedMass)
                    assert reduced_mass in rounded_reducedMass, f"The reduced mass in the molscat output should be an element of {self}.reducedMass."
                    reduced_mass_index = rounded_reducedMass.index(reduced_mass)

                # determine which basis set is used in the output
                elif "in a total angular momentum basis" in line:
                    tcpld = True
                    basis = ('L', 'F1', 'F2', 'F12', 'T', 'MT')
                    self.Qn = qn.Tcpld
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
                    rounded_singletParameter = tuple(round(singlet_parameter, sigfigs = 12) for singlet_parameter in self.singletParameter)
                    assert A_s in rounded_singletParameter, f"The singlet scaling parameter from the molscat output should be an element of {self}.singletParameter."
                    A_s_index = rounded_singletParameter.index(A_s)

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
                    rounded_tripletParameter = tuple(round(triplet_parameter, sigfigs = 12) for triplet_parameter in self.tripletParameter)
                    assert A_t in rounded_tripletParameter, f"The triplet scaling parameter from the molscat output should be an element of {self}.tripletParameter."
                    A_t_index = rounded_tripletParameter.index(A_t)

                elif "INPUT ENERGY LIST IS" in line:
                    while "CALCULATIONS WILL BE PERFORMED FOR" not in line:
                        line = next(molscat_output)
                    line = next(molscat_output)
                    # create the list of energies from the output
                    energy_list = []
                    # append each energy value from the output to the list of energies
                    while line.strip():
                        # the energies in the molscat outputs are listed in G17.10 format here anyway
                        energy_list.append(round(float(line.split()[6]), sigfigs = 11))
                        line = next(molscat_output)
                    energy_tuple = tuple(energy_list)

                    if self.collisionEnergy == None: self.collisionEnergy = energy_tuple
                    rounded_collisionEnergy = tuple(round(energy, sigfigs = 11) for energy in self.collisionEnergy )
                    assert energy_tuple == rounded_collisionEnergy, f"The list of collision energies from the molscat output should be equal to {self}.collisionEnergy."

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
                    
                    S = SMatrix(basis = basis, diagonal = self.diagonal, C4 = self.C4[C4_index], singletParameter = self.singletParameter[A_s_index], tripletParameter = self.tripletParameter[A_t_index], reducedMass = self.reducedMass[reduced_mass_index], magneticField = self.magneticField[magnetic_field_index], collisionEnergy = self.collisionEnergy[energy_counter], matrix = matrix)
                    if (C4_index, A_s_index, A_t_index, reduced_mass_index, magnetic_field_index, energy_counter) in self.matrixCollection.keys():
                        self.matrixCollection[CollectionParametersIndices(C4_index, A_s_index, A_t_index, reduced_mass_index, magnetic_field_index, energy_counter)].update(S)
                    else:
                        self.matrixCollection[CollectionParametersIndices(C4_index, A_s_index, A_t_index, reduced_mass_index, magnetic_field_index, energy_counter)] = S
                    energy_counter +=1
            
            del self.Qn
            

    @classmethod
    def from_output(cls, file_path: str):
        """Create an SMatrixCollection object from a molscat output.

        :param file_path: path to the file containing molscat output.
        :return: SMatrixCollection object.
        """
        s_collection = SMatrixCollection()
        s_collection.update_from_output(file_path = file_path)
        return s_collection


    def encodableMatrixCollection(self) -> dict:      
        encodable_matrix_collection = { str(tuple(key)): value.encodableForm() for key, value in self.matrixCollection.items() }
        return encodable_matrix_collection


    def encodableForm(self) -> dict:
        attributes = vars(self).copy()
        attributes.pop('matrixCollection')
        
        for attr, value in attributes.items():
            if isinstance(value, tuple):
                attributes[attr] = str(value)

        encodable_form = { "__SMatrixCollection__": True }
        encodable_form.update( self.__class__(**attributes, matrixCollection = self.encodableMatrixCollection() ).__dict__ )
        
        return encodable_form

    @staticmethod
    def default(o):
        if isinstance(o, complex):
            return { '__complex__': True, 'abs': abs(o), 'phase': cmath.phase(o) }
        elif isinstance(o, SMatrix) or isinstance(o, SMatrixCollection):
            return o.encodableForm()
        return o


    def toJSON(self, file_path: str) -> None:
        with open(file_path, 'w') as file:
            json.dump(self, file, default = self.default, indent = 3)


    def toMsgPack(self, file_path: str) -> None:
        with open(file_path, 'wb') as file:
            msgpack.pack(self, file, default = self.default)


    def toPickle(self, file_path: str) -> None:
        with open(file_path, 'wb') as file:
            pickle.dump(self, file, protocol = 5)


    @classmethod
    def decode(cls, dct):
        if '__complex__' in dct:
            return cmath.rect(dct['abs'], dct['phase'])

        elif '__SMatrix__' in dct:
            return SMatrix.decode(dct)

        elif '__SMatrixCollection__' in dct:
            matrix_collection = dct.get('matrixCollection')

            ## parsing tuples of parameters and basis name
            for key, value in dct.items():
                if isinstance(value, str) and value.startswith('(') and value.endswith(')'):
                    dct[key] = eval(value)


            new_matrix_collection = { CollectionParametersIndices( *eval(key) ): value for key, value in matrix_collection.items() }
            
            dct.pop('matrixCollection')
            dct.pop('__SMatrixCollection__')

            return cls(**dct, matrixCollection = new_matrix_collection)
        
        return dct


    @classmethod
    def fromJSON(cls, file_path):
        with open(file_path, 'r') as file:
            s_matrix_collection = json.load(file, object_hook = cls.decode)
            return s_matrix_collection


    @classmethod
    def fromMsgPack(cls, file_path):
        with open(file_path, 'rb') as file:
            s_matrix_collection = msgpack.unpack(file, object_hook = cls.decode)
            return s_matrix_collection


    @classmethod
    def fromPickle(cls, file_path):
        with open(file_path, 'rb') as file:
            s_matrix_collection = pickle.load(file)
            return s_matrix_collection


    def getParamIndicesAsArray(self, **kwargs) -> CollectionParametersIndices[tuple[int, ...], ...]:
        """Get the indices of the parameters from a dictionary.
        
        \*\*kwargs:
        :param dict param_indices: dictionary with entries of the form
        'parameter_name: tuple(parameter_indices)'
        :param dict param_values: dictionary with entries of the form
        'parameter_name: tuple(parameter_values)'. Ignored if
        param_indices is provided.
        :return: CollectionParametersIndices object containing tuples of
        parameter indices specified in param_indices or param_values, or
        all indices for the parameters if they weren't mentioned in the
        arguments.

        """
        param_indices = kwargs['param_indices'] if 'param_indices' in kwargs.keys() else None
        param_values = kwargs['param_values'] if 'param_values' in kwargs.keys() else None

        if param_indices == None and param_values == None:
            param_indices = CollectionParametersIndices(*(range( len(getattr(self, attr_name) ) ) for attr_name in CollectionParametersIndices._fields) )

        elif param_indices == None:           
            if not param_values.keys() <= set(CollectionParametersIndices._fields): raise AttributeError((f"param_values keys can only include the attributes of CollectionParametersIndices objects."))
            try:
                param_indices = CollectionParametersIndices( *( tuple(getattr(self, attr_name).index(x) for x in param_values.get(attr_name) ) if attr_name in param_values.keys() else range(len(getattr(self, attr_name))) for attr_name in CollectionParametersIndices._fields ) )
            except ValueError:
                not_present_values = { attr_name: tuple(value for value in param_values[attr_name] if value not in getattr(self, attr_name)) for attr_name in param_values.keys() if not set(param_values[attr_name]).issubset(getattr(self, attr_name)) }
                raise ValueError(f"The following parameter values: {not_present_values} are not present in the collection.")

        else:
            param_indices = CollectionParametersIndices( *( param_indices.get(attr_name, range(len(getattr(self, attr_name)))) for attr_name in CollectionParametersIndices._fields ) )

        return param_indices


    def getAsArray(self, qn_out: qn.LF1F2 | qn.LF12 | qn.Tcpld, qn_in: qn.LF1F2 | qn.LF12 | qn.Tcpld, **kwargs) -> np.ndarray[Any, complex]:
        """Get S-matrix elements as an array for the given parameter values.

        :param qn_out: quantum numbers for the final state.
        :param qn_in: quantum numbers for the initial state.
        :param param_indices: dictionary with entries of the form
        'parameter_name: tuple(parameter_indices)'
        :param param_values: dictionary with entries of the form
        'parameter_name: tuple(parameter_values)'. Ignored if
        param_indices is provided.
        :return: CollectionParametersIndices object containing tuples of
        parameter indices specified in param_indices or param_values, or
        all indices for the parameters if they weren't mentioned in the
        arguments.
        """

        param_indices = self.getParamIndicesAsArray(**kwargs)
        S_array = np.fromiter( ( self.matrixCollection[CollectionParametersIndices(*indices_combination)].getInBasis(qn_out, qn_in) for indices_combination in itertools.product( *param_indices )), dtype = complex).reshape( *(len(index_tuple) for index_tuple in param_indices) )
        
        return S_array


    def getCrossSection(self, qn_out: qn.LF1F2 | qn.LF12 | qn.Tcpld, qn_in: qn.LF1F2 | qn.LF12 | qn.Tcpld, **kwargs) -> np.ndarray[Any, float]:
        """Get the energy-dependent cross section for the given final
          and initial state as an array for the given parameter values.

        :param qn_out: quantum numbers for the final state.
        :param qn_in: quantum numbers for the initial state.

        \*\*kwargs"
        :param param_indices: dictionary with entries of the form
        'parameter_name: tuple(parameter_indices)'
        :param param_values: dictionary with entries of the form
        'parameter_name: tuple(parameter_values)'. Ignored if
        param_indices is provided.

        :return: array consisting of the cross sections for
          the parameters given in **kwargs (or all parameter values
            in the collection, if they weren't specified).
        """

        param_indices = self.getParamIndicesAsArray(**kwargs)
        cross_section_array = np.fromiter( ( self.matrixCollection[CollectionParametersIndices(*indices_combination)].getCrossSection(qn_out, qn_in) for indices_combination in itertools.product( *param_indices )), dtype = float).reshape( *(len(index_tuple) for index_tuple in param_indices) )
        return cross_section_array


    def getMomentumTransferCrossSection(self, qn_in: qn.LF1F2 | qn.LF12 | qn.Tcpld, **kwargs) -> np.ndarray[Any, float]:
        """Get the energy-dependent momentum-transfer cross section for 
        the given initial state as an array for the given parameter values.

        :param qn_in: quantum numbers for the initial state.

        \*\*kwargs"
        :param param_indices: dictionary with entries of the form
        'parameter_name: tuple(parameter_indices)'
        :param param_values: dictionary with entries of the form
        'parameter_name: tuple(parameter_values)'. Ignored if
        param_indices is provided.
        
        :return: array consisting of the cross sections for
          the parameters given in **kwargs (or all parameter values
            in the collection, if they weren't specified).
        """

        param_indices = self.getParamIndicesAsArray(**kwargs)
        cross_section_array = np.fromiter( ( self.matrixCollection[CollectionParametersIndices(*indices_combination)].getMomentumTransferCrossSection(qn_in) for indices_combination in itertools.product( *param_indices )), dtype = float).reshape( *(len(index_tuple) for index_tuple in param_indices) )
        return cross_section_array


    def getRateCoefficient(self, qn_out: qn.LF1F2 | qn.LF12 | qn.Tcpld, qn_in: qn.LF1F2 | qn.LF12 | qn.Tcpld, unit = None, **kwargs) -> np.ndarray[Any, float]:
        """Get the energy-dependent rate coefficient for the given final
          and initial state as an array for the given parameter values.

        :param qn_out: quantum numbers for the final state.
        :param qn_in: quantum numbers for the initial state.

        \*\*kwargs"
        :param param_indices: dictionary with entries of the form
        'parameter_name: tuple(parameter_indices)'
        :param param_values: dictionary with entries of the form
        'parameter_name: tuple(parameter_values)'. Ignored if
        param_indices is provided.
        
        :return: array consisting of the rate coefficients for
          the parameters given in **kwargs (or all parameter values
            in the collection, if they weren't specified).
        """

        param_indices = self.getParamIndicesAsArray(**kwargs)
        rate_coefficient_array = np.fromiter( ( self.matrixCollection[CollectionParametersIndices(*indices_combination)].getRateCoefficient(qn_out, qn_in, unit = unit) for indices_combination in itertools.product( *param_indices )), dtype = float).reshape( *(len(index_tuple) for index_tuple in param_indices) )
        return rate_coefficient_array


    def getMomentumTransferRateCoefficient(self, qn_in: qn.LF1F2 | qn.LF12 | qn.Tcpld, unit = None, **kwargs) -> np.ndarray[Any, float]:
        """Get the energy-dependent momentum-transfer rate coefficient for
          the given initial state as an array for the given parameter values.

        :param qn_in: quantum numbers for the initial state.

        \*\*kwargs"
        :param param_indices: dictionary with entries of the form
        'parameter_name: tuple(parameter_indices)'
        :param param_values: dictionary with entries of the form
        'parameter_name: tuple(parameter_values)'. Ignored if
        param_indices is provided.
        
        :return: array consisting of the rate coefficients for
          the parameters given in **kwargs (or all parameter values
            in the collection, if they weren't specified).
        """

        param_indices = self.getParamIndicesAsArray(**kwargs)
        rate_coefficient_array = np.fromiter( ( self.matrixCollection[CollectionParametersIndices(*indices_combination)].getMomentumTransferRateCoefficient(qn_in, unit = unit) for indices_combination in itertools.product( *param_indices )), dtype = float).reshape( *(len(index_tuple) for index_tuple in param_indices) )
        return rate_coefficient_array


    def thermalAverage(self, array_to_average: np.ndarray[Any, float], distribution_iterator: Iterable = None) -> np.ndarray[Any, float]:
        """Thermally average an array of values.

        :param array_to_average: array of energy-depending values
          in the last axis.
        :param distribution_iterator: an iterable object providing
          the distribution factors in the integral.
        :return: array of the thermally averaged values,
          with the last axis contracted with respect to array_to_average.
        """


        if distribution_iterator == None:
            distribution_iterator = n_root_iterator(temperature = 5e-4, E_min = min(self.collisionEnergy), E_max = max(self.collisionEnergy), N = len(self.collisionEnergy), n = 3)

        distribution_array = np.fromiter( distribution_iterator, dtype = float )
        integrand = array_to_average * distribution_array
        integral = scipy.integrate.simpson( integrand )
        norm = scipy.integrate.simpson( distribution_array )

        averaged_array = integral / norm

        return averaged_array


    def getThermallyAveragedRate(self, qn_out: qn.LF1F2 | qn.LF12 | qn.Tcpld, qn_in: qn.LF1F2 | qn.LF12 | qn.Tcpld, distribution_iterator: Iterable = None, unit: str = None, **kwargs) -> np.ndarray[Any, float]:
        """Get the thermally averaged rate coefficient for the given final
          and initial state as an array for the given parameter values.

        :param qn_out: quantum numbers for the final state.
        :param qn_in: quantum numbers for the initial state.
        :param distribution_iterator: an iterable object providing
          the distribution factors in the integral.

        \*\*kwargs"
        :param param_indices: dictionary with entries of the form
        'parameter_name: tuple(parameter_indices)'
        :param param_values: dictionary with entries of the form
        'parameter_name: tuple(parameter_values)'. Ignored if
        param_indices is provided.
        
        :return: array consisting of the thermal rate coefficients for
          the parameters given in **kwargs (or all parameter values
            in the collection, if they weren't specified).
        """       

        rate_array_to_average = self.getRateCoefficient(qn_out, qn_in, unit, **kwargs)
        averaged_rate_array = self.thermalAverage(rate_array_to_average, distribution_iterator)
        return averaged_rate_array


    def getThermallyAveragedMomentumTransferRate(self, qn_in: qn.LF1F2 | qn.LF12 | qn.Tcpld, distribution_iterator: Iterable = None, unit: str = None, **kwargs) -> np.ndarray[Any, float]:
        """Get the thermally averaged momentum-transfer rate coefficient
          for the given final and initial state as an array
            for the given parameter values.

        :param qn_in: quantum numbers for the initial state.
        :param distribution_iterator: an iterable object providing
          the distribution factors in the integral.

        \*\*kwargs"
        :param param_indices: dictionary with entries of the form
        'parameter_name: tuple(parameter_indices)'
        :param param_values: dictionary with entries of the form
        'parameter_name: tuple(parameter_values)'. Ignored if
        param_indices is provided.
        
        :return: array consisting of the thermal momentum-transfer rate
          coefficients for the parameters given in **kwargs (or all
            parameter values in the collection, if they weren't specified).
        """   

        rate_array_to_average = self.getMomentumTransferRateCoefficient(qn_in, unit, **kwargs)
        averaged_rate_array = self.thermalAverage(rate_array_to_average, distribution_iterator)
        return averaged_rate_array


    def getProbability(self, qn_out: qn.LF1F2 | qn.LF12 | qn.Tcpld, qn_in: qn.LF1F2 | qn.LF12 | qn.Tcpld, qn_momentum_transfer: qn.LF1F2 | qn.LF12 | qn.Tcpld = None, **kwargs) -> np.ndarray[Any, float]:
        """Get the energy-dependent probability of collision calculated
          as the ratio of the rate coefficient for the transition from the 
          given initial to the final state, and the momentum-transfer
            rate coeffient.

        :param qn_out: quantum numbers for the final state.
        :param qn_in: quantum numbers for the initial state.
        :param qn_momentum_transfer: quantum numbers for the initial state
          for calculating the momentum-transfer rate coefficient.

        \*\*kwargs"
        :param param_indices: dictionary with entries of the form
        'parameter_name: tuple(parameter_indices)'
        :param param_values: dictionary with entries of the form
        'parameter_name: tuple(parameter_values)'. Ignored if
        param_indices is provided.
        
        :return: array consisting of the energy-dependent probability
          for the parameters given in **kwargs (or all parameter values
            in the collection, if they weren't specified).
        """           

        if qn_momentum_transfer == None:
            qn_momentum_transfer = qn_in
        
        rate_array = self.getRateCoefficient(qn_out, qn_in, **kwargs)
        momentum_transfer_rate_array = self.getMomentumTransferRateCoefficient(qn_momentum_transfer, **kwargs)
        probability_array = rate_array / momentum_transfer_rate_array

        return probability_array


    def getThermallyAveragedProbability(self, qn_out: qn.LF1F2 | qn.LF12 | qn.Tcpld, qn_in: qn.LF1F2 | qn.LF12 | qn.Tcpld, qn_momentum_transfer: qn.LF1F2 | qn.LF12 | qn.Tcpld = None, distribution_iterator: Iterable = None, **kwargs) -> np.ndarray[Any, float]:
        """Get the thermally averaged probability of collision calculated
          as the ratio of the energy-dependent rate coefficient for the
            transition from the given initial to the final state, and
              the momentum-transfer rate coeffient.

        :param qn_out: quantum numbers for the final state.
        :param qn_in: quantum numbers for the initial state.
        :param qn_momentum_transfer: quantum numbers for the initial state
          for calculating the momentum-transfer rate coefficient.
        :param distribution_iterator: an iterable object providing
          the distribution factors in the integral.

        \*\*kwargs"
        :param param_indices: dictionary with entries of the form
        'parameter_name: tuple(parameter_indices)'
        :param param_values: dictionary with entries of the form
        'parameter_name: tuple(parameter_values)'. Ignored if
        param_indices is provided.
        
        :return: array consisting of the thermally averaged probability
          for the parameters given in **kwargs (or all parameter values
            in the collection, if they weren't specified).
        """ 

        probability_array_to_average = self.getProbability(qn_out, qn_in, qn_momentum_transfer, **kwargs)
        averaged_probability_array = self.thermalAverage(probability_array_to_average, distribution_iterator)
        return averaged_probability_array


    def getProbabilityFromThermalAverages(self, qn_out: qn.LF1F2 | qn.LF12 | qn.Tcpld, qn_in: qn.LF1F2 | qn.LF12 | qn.Tcpld, qn_momentum_transfer: qn.LF1F2 | qn.LF12 | qn.Tcpld = None, distribution_iterator: Iterable = None, **kwargs) -> np.ndarray[Any, float]:
        """Get the probability of collision calculated as the ratio of the
          thermally averaged rate coefficient for the transition from
            the given initial to the final state, and the momentum-transfer
              rate coefficient.

        :param qn_out: quantum numbers for the final state.
        :param qn_in: quantum numbers for the initial state.
        :param qn_momentum_transfer: quantum numbers for the initial state
          for calculating the momentum-transfer rate coefficient.
        :param distribution_iterator: an iterable object providing
          the distribution factors in the integral.

        \*\*kwargs"
        :param param_indices: dictionary with entries of the form
        'parameter_name: tuple(parameter_indices)'
        :param param_values: dictionary with entries of the form
        'parameter_name: tuple(parameter_values)'. Ignored if
        param_indices is provided.
        
        :return: array consisting of the thermally averaged probability
          for the parameters given in **kwargs (or all parameter values
            in the collection, if they weren't specified).
        """ 

        averaged_rate_array = self.getThermallyAveragedRate(qn_out, qn_in, distribution_iterator, **kwargs)
        averaged_momentum_transfer_rate_array = self.getThermallyAveragedMomentumTransferRate(qn_momentum_transfer, **kwargs)
        probability_from_averages_array = averaged_rate_array / averaged_momentum_transfer_rate_array
        return probability_from_averages_array




def summing(MF_in, MS_in, L_max, smatrix):
    time_0 = time.perf_counter()

    # Lmax = 2*9
    
    x = [sum([
            sum([
                    0.1*np.abs(smatrix.matrixCollection[0,0,0,0,0,i].getInBasis(qn.LF1F2(L, ML, 2, MF, 1, MS), qn.LF1F2(L, ML, 4, MF_in, 1, MS_in)))**2 for i in range(10)
                ])
                for ML in range(-L, L+1, 2)
            ])
       for MF in range(-2, 2+1, 2) for MS in range(-1, 1+1, 2) for L in range(0, L_max+1, 2)]
    

    return (MF_in, MS_in), sum(x), time.perf_counter()-time_0

def main():
    from multiprocessing import Pool
    from multiprocessing.dummy import Pool as ThreadPool    
    
    def json_dump_load():
        time_0 = time.perf_counter()
        s = SMatrixCollection.from_output(r"../data/TEST_10_ENERGIES.output")
        s.toJSON(r"../data_produced/json_test_3.json")

        s2 = SMatrixCollection.fromJSON(r"../data_produced/json_test_3.json")
        print(s2)
        print(s == s2)
        print(f"Loading the matrix took {time.perf_counter() - time_0} seconds.")
    
    def msgpack_pack_unpack():
        time_0 = time.perf_counter()
        s = SMatrixCollection.from_output(r"../data/TEST_10_ENERGIES.output")
        s.toMsgPack(r"../data_produced/json_test_3.msgpack")

        s2 = SMatrixCollection.fromMsgPack(r"../data_produced/json_test_3.msgpack")
        print(s2)
        print(s == s2)
        print(s2.matrixCollection == s.matrixCollection)
        x = s.matrixCollection[(0,0,0,0,0,0,)]
        y = s2.matrixCollection[(0,0,0,0,0,0,)]
        print( y == x )
        print( y.matrix == x.matrix )

        difference = { key: y.matrix[key] - x.matrix[key] for key in y.matrix.keys() if (abs(y.matrix[key] - x.matrix[key]) > 1e-16) }
        print(difference)
        ### floating point precisions problems, something to test in tests
        difference = { key: y.matrix[key] - x.matrix[key] for key in x.matrix.keys() if (abs(y.matrix[key] - x.matrix[key]) > 1e-16) }
        print(difference)
        print(f"Loading the matrix took {time.perf_counter() - time_0} seconds.")

    def pickle_dump_load():
        time_0 = time.perf_counter()
        s = SMatrixCollection.from_output(r"../data/TEST_10_ENERGIES.output")
        s.toPickle(r"../data_produced/json_test_3.pickle")

        s2 = SMatrixCollection.fromPickle(r"../data_produced/json_test_3.pickle")
        print(s2)
        print(s == s2)
        print(s2.matrixCollection == s.matrixCollection)
        x = s.matrixCollection[(0,0,0,0,0,0,)]
        y = s2.matrixCollection[(0,0,0,0,0,0,)]
        print( y == x )
        print( y.matrix == x.matrix )

        difference = { key: y.matrix[key] - x.matrix[key] for key in y.matrix.keys() if (abs(y.matrix[key] - x.matrix[key]) > 1e-16) }
        print(difference)
        ### floating point precisions problems, something to test in tests
        difference = { key: y.matrix[key] - x.matrix[key] for key in x.matrix.keys() if (abs(y.matrix[key] - x.matrix[key]) > 1e-16) }
        print(difference)
        print(f"Loading the matrix took {time.perf_counter() - time_0} seconds.")

    def unpickle_json_dump():
        time_0 = time.perf_counter()
        s = SMatrixCollection.fromPickle(r"../data_produced/json_test_3.pickle")
        x = len( s.matrixCollection[(0,0,0,0,0,0,)].matrix )
        print(x)
        s.toJSON(r"../data_produced/pickle_to_json_test_3.json")
        print(f"Loading the matrix took {time.perf_counter() - time_0} seconds.")

    # msgpack_pack_unpack()
    # pickle_dump_load()
    unpickle_json_dump()
    # time_0 = time.perf_counter()
    # x = s.matrixCollection[(0,0,0,0,0,0)]
    # print(x)
    # y = json.dumps(x.encodableForm(), cls = SMatrixCollectionEncoder)
    # print(y)
    # z = json.loads(y, object_hook = decode_smatrix)
    # print(z)
    # print(z == x)

    # print(z.matrix == x.matrix)
    # difference = { key: z.matrix[key] - x.matrix[key] for key in z.matrix.keys() if (abs(z.matrix[key] - x.matrix[key]) > 1e-15) }
    # print(difference)
    # ### floating point precisions problems, something to test in tests
    # difference = { key: x.matrix[key] - z.matrix[key] for key in x.matrix.keys() if (abs(x.matrix[key] - z.matrix[key]) > 1e-15) }
    # print(difference)
    # print(set(z.matrix.items()).difference(set(x.matrix.items())))
    # print(set(x.matrix.items()).difference(set(z.matrix.items())))

    # x = set({"a": 231, "b": 321321.2}.items())
    # y = set({"a": 1.2, "c": 2121 }.items())
    # print(x.difference(y))
    # ss = s.decodableForm()
    # print(ss)
    # s.toJSON(r"../data_produced/json_test_2.json")
    # ss = s.decodable_matrix_collection()
    # print(ss)
    # with open(r"../data_produced/test_json.json", 'w') as file:
    #     json.dump(ss, file, cls = ComplexEncoder)
    # print(s.matrixCollection[(0,0,0,0,0,0)].matrix)
    # print(f"Loading the matrix took {time.perf_counter() - time_0} seconds.")

    L_max = 2*9

    # args = ((MF,MS, L_max, s) for MF in range(-4,4+1,2) for MS in range(-1, 1+1, 2))
    def multiprocessing_test(args):
        time_0 = time.perf_counter()
        with Pool() as pool:
            results = pool.starmap(summing, args)
            for arg, result, duration in results:
                # result, duration = summing(*arg)
                print(f"The sum for (MF, MS) = {arg} is {result}. It took {duration:.2f} s to sum up {10*2*3*(L_max/2+1)**2:.2e} elements, on average {duration/(10*2*3*(L_max/2+1)**2) :.2e} s per element.")
        
        duration = time.perf_counter()-time_0
        print(f"The total time was {duration:.2f} s for summing up {5*2*10*2*3*(L_max/2+1)**2:.2e} elements, on average {duration/(5*2*10*2*3*(L_max/2+1)**2):.2e} s per element.")

    def simple_loop_test(args):
        time_0 = time.perf_counter()
        for arg in args:
            _, result, duration = summing(*arg)
            print(f"The sum for (MF, MS) = {_} is {result}. It took {duration:.2f} s to sum up {10*2*3*(L_max/2+1)**2:.2e} elements, on average {duration/(10*2*3*(L_max/2+1)**2) :.2e} s per element.")
        
        duration = time.perf_counter()-time_0
        print(f"The total time was {duration:.2f} s for summing up {5*2*10*2*3*(L_max/2+1)**2:.2e} elements, on average {duration/(5*2*10*2*3*(L_max/2+1)**2):.2e} s per element.")

    def vectorize_test(smatrix):
        L = 2
        lst_out = [ ML for ML in range(-L, L+1, 2)]
        lst_in = [ ML for ML in range(-L, L+1, 2)]
        lst_out, lst_in = np.meshgrid(lst_out, lst_in)
        # basis = [('L', 'ML', 'F1', 'MF1', 'F2', 'MF2') for ML in range(-L, L+1, 2)]
        f = lambda ML_out, ML_in: smatrix.matrixCollection[0,0,0,0,0,0].getInBasis(qn.LF1F2(L, ML_out, 2, 0, 1, -1), qn.LF1F2(L, ML_in, 4, 0, 1, 1))
        g = np.vectorize(f)
        return np.abs(g(lst_out, lst_in))**2, lst_out, lst_in

    # args = ((MF,MS, L_max, s) for MF in range(-4,4+1,2) for MS in range(-1, 1+1, 2))
    # multiprocessing_test(args)
    # args = ((MF,MS, L_max, s) for MF in range(-4,4+1,2) for MS in range(-1, 1+1, 2))
    # simple_loop_test(args)
    # x, lst_out, lst_in = vectorize_test(s)
    # print(x, '\n', lst_out, '\n', lst_in)
    # x = s.matrixCollection[0,0,0,0,0,0].getInBasis(qn.LF1F2(2, -2, 4, 0, 1, 1), qn.LF1F2(2, -2, 2, 0, 1, 1))
    # L_max, nenergies = 2*29, 1
    # qns = [(qn.LF1F2(L, ML, 2, 0, 1, 1), qn.LF1F2(L, ML, 4, 0, 1, 1)) for L in range (0, L_max+1, 2) for ML in range(-L, L+1, 2)]
    # # x = sum([s.matrixCollection[0,0,0,0,0,8].getCrossSection(*qns) for qns in qns])
    # # print(x)
    # time_0 = time.perf_counter()
    # x = sum([(1/nenergies)*s.matrixCollection[0,0,0,0,0,i].getRateCoefficient(*qns, unit = 'cm**3/s') for qns in qns for i in range(nenergies)])
    # duration = time.perf_counter()-time_0
    # print(f"The time was {duration:.2e} s.")
    # print(x)
    
    # time_0 = time.perf_counter()
    # lst0 = [s.matrixCollection[0,0,0,0,0,i].getMomentumTransferRateCoefficient(qn.LF1F2(None, 0, 4, -4, 1, -1), unit = 'cm**3/s') for i in range(10) ]
    # lst1 = [s.matrixCollection[0,0,0,0,0,i].getMomentumTransferRateCoefficient(qn.LF12(None, 0, 4, 1, 5, -5), unit = 'cm**3/s') for i in range(10)]
    # print(lst0)
    # print(lst1)
    # duration = time.perf_counter()-time_0
    # print(f"The time was {duration:.2e} s.")
    # print(s.getAsArray(0,0))
    # print(s.getAsArray(0,0, param_values = {'C4': (159.9,)}))
    # print(s.getAsArray(0,0, param_indices = {'C4': (0,), 'collisionEnergy': (1,3,5,7,9)}))
    # x = s.getAsArray(qn.LF1F2(0,0,2,0,1,1), qn.LF1F2(0,0,4,0,1,1), param_indices = {'C4': (0,), 'collisionEnergy': (1,3,5,7,9)})
    # print(x)
    # y = s.getAsArray(qn.LF1F2(0,0,2,0,1,1), qn.LF1F2(0,0,4,0,1,1), param_values = {'C4': (159.9,), 'collisionEnergy': (1e-6, 1e-5, 1e-4, 5e-4, 1e-3, 5e-3)})
    # print(y)
    # z = s.getRateCoefficientAsArray(qn.LF1F2(0,0,2,0,1,1), qn.LF1F2(0,0,4,0,1,1), param_values = {'C4': (159.9,), 'collisionEnergy': (1e-6, 5e-6, 1e-5, 1e-4, 5e-4, 1e-3)})
    # print(z*10**6*rate_from_au_to_SI)
    # zz = s.getRateCoefficientAsArray(qn.LF1F2(0,0,2,0,1,1), qn.LF1F2(0,0,4,0,1,1), param_values = {'C4': (159.9,), 'collisionEnergy': (1e-6, 5e-6, 1e-5, 1e-4, 5e-4, 1e-3)}, unit = 'cm**3/s')
    # time_0 = time.perf_counter()
    # avv = s.getThermallyAveragedRate(qn.LF1F2(0,0,2,0,1,1), qn.LF1F2(0,0,4,0,1,1))
    # print(avv)
    # duration = time.perf_counter()-time_0
    # print(f"The time was {duration:.2e} s.")    

    # time_0 = time.perf_counter()
    # # f = lambda F_out, MF_out, MS_out, F_in, MF_in, MS_in: sum(  s.getThermallyAveragedRateAsArray(qn.LF1F2(L, ML, F1 = F_out, MF1 = MF_out, F2 = 1, MF2 = MS_out), qn.LF1F2(L, ML, F1 = F_in, MF1 = MF_in, F2 = 1, MF2 = MS_in)) for L in range(0, 58+1, 2) for ML in range(-L, L+1, 2))
    # # x = f(2, 0, -1, 4, -2, 1)
    # # print(x)

    # iterrr = ( s.getRateCoefficientAsArray(qn.LF1F2(L, ML, F1 = F_out, MF1 = MF_out, F2 = 1, MF2 = MS_out), qn.LF1F2(L, ML, F1 = F_in, MF1 = MF_in, F2 = 1, MF2 = MS_in)) for L in range(0, 58+1, 2) for ML in range(-L, L+1, 2) )
    # print(type(iter))
    # f = lambda : np.fromiter(  iterr, dtype = float).sum()
    # def f(F_out, MF_out, MS_out, F_in, MF_in, MS_in):
    #     x = sum( s.getRateCoefficient(qn.LF1F2(L, ML, F1 = F_out, MF1 = MF_out, F2 = 1, MF2 = MS_out), qn.LF1F2(L, ML, F1 = F_in, MF1 = MF_in, F2 = 1, MF2 = MS_in), unit = 'cm**3/s') for L in range(0, 18+1, 2) for ML in range(-L, L+1, 2) )
    #     return s.thermalAverage(x)
    # x = f(2, 0, -1, 4, -2, 1)
    # x = s.thermalAverage(x)
    # print(x, type(x))
    # duration = time.perf_counter()-time_0
    # print(f"The time was {duration:.2e} s.") 

    # time_0 = time.perf_counter()
    # g = np.vectorize(f, signature = '(),(),(),(),(),() -> (a,b,c,d,e)' )
    # F_in = 4
    # MF_in = np.arange(-F_in, F_in+1, 2)
    # # MF_in = 0
    # # MF_in = [0, 2]
    # S = 1
    # MS_in = np.arange(-S, S+1, 2)
    # # MS_in = 1
    # F_out = 2
    # MF_out = np.arange(-F_out, F_out+1, 2)
    # # MF_out = 0
    # # MF_out = [0, 2]
    # MS_out = np.arange(-S, S+1, 2)
    # # MS_out = -1
    # MF_out, MS_out, MF_in, MS_in = np.meshgrid(MF_out, MS_out, MF_in, MS_in)
    # # MF_out, MF_in = np.meshgrid(MF_out, MF_in)
    # print(MF_out)
    # print(MS_out)
    # print(MF_in)
    # print(MS_in)
    # y = g(F_out, MF_out, MS_out, F_in, MF_in, MS_in)
    # y = g( 2, 0, -1, 4, 0, -1 )

    # print( y.squeeze(), type(y))
    # duration = time.perf_counter()-time_0
    # print(f"The time was {duration:.2e} s.")   

    # x = s.getRateCoefficientAsArray(qn.LF1F2(4, 2, 2, 0, 1, -1), qn.LF1F2(4, 0, 4, 0, 1, 1), unit = 'cm**3/s')
    # print(x)

    # x = s.getRateCoefficientAsArray(qn.LF1F2(4, 0, 2, 2, 1, -1), qn.LF1F2(4, 0, 4, 0, 1, 1), unit = 'cm**3/s')
    # print(x)


    # print(s.matrixCollection[0,0,0,0,0,0].matrix)

if __name__ == '__main__':
    main()