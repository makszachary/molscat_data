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
from .analytical import allThresholds


class CollectionParameters(NamedTuple):
    C4: float | tuple[float, ...]
    singletParameter: float | tuple[float, ...]
    tripletParameter: float | tuple[float, ...]
    spinOrbitParameter: float | tuple[float, ...]
    reducedMass: float | tuple[float, ...]
    magneticField: float | tuple[float, ...]
    collisionEnergy: float | tuple[float, ...]


class CollectionParametersIndices(NamedTuple):
    C4: int | tuple[int, ...]
    singletParameter: int | tuple[int, ...]
    tripletParameter: int | tuple[int, ...]
    spinOrbitParameter: int | tuple[int, ...]
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
    spinOrbitParameter: float = None
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

    def getMomentumTransferCrossSectionVsL(self, qn_in: tuple | qn.LF1F2 | qn.LF12 | qn.Tcpld, basis: str = None) -> float:
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
        print(f'{L_max = }', flush = True)
        print(f'{phase_shift = }', flush = True)
        print(f'{cross_section = }', flush = True)
        return cross_section

    def getMomentumTransferCrossSection(self, qn_in: tuple | qn.LF1F2 | qn.LF12 | qn.Tcpld, basis: str = None) -> float:
        """Calculate the momentum-transfer cross section in the given basis,
        summed over all the partial waves.
        
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

    def getMomentumTransferRateCoefficientVsL(self, qn_in: tuple | qn.LF1F2 | qn.LF12 | qn.Tcpld, basis: str = None, unit : str = None) -> float:
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

        rate_coefficient = np.sqrt(2*(self.collisionEnergy/Hartree_to_K)/self.reducedMass) * self.getMomentumTransferCrossSectionVsL(**arguments)
        
        match unit:
            case 'cm**3/s':
                rate_coefficient *= 10**6 * rate_from_au_to_SI
            case None | 'a.u.':
                pass
            case _:
                raise ValueError(f"The possible values of unit are: 'cm**3/s', 'a.u.', None (default), {unit=} matching none of these.")

        print(f'{rate_coefficient = }', flush = True)
        print(f'{rate_coefficient.shape = }', flush = True)
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

    C4: tuple[float, ...] = (None,)
    singletParameter: tuple[float, ...] = (None,)
    tripletParameter: tuple[float, ...] = (None,)
    spinOrbitParameter: tuple[float, ...] = (None,)
    reducedMass: tuple[float, ...] = (None,)
    magneticField: tuple[float, ...] = (None,)
    collisionEnergy: tuple[float, ...] = (None,)

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


    @staticmethod
    def get_channels_fmfbasis(line):
        pass

        return

    def update_from_output(self, file_path: str, non_molscat_so_parameter = None):
        """Update the S-matrix collection with data from a single molscat output file in tcpld basis.

        :param file_path: Path to the molscat output file.
        :param non_molscat_so_parameter: scaling of the spin-orbit coupling
        if the data was scaled before putting into RKHS routine.
        """
        
        with open(file_path,'r') as molscat_output:
            nextra = 0
            for line in molscat_output:
                if "REDUCED MASS FOR INTERACTION =" in line:
                    # reduced mass is given F14.9 format in amu in MOLSCAT output, we convert it to atomic units and round
                    reduced_mass = round(float(line.split()[5])*amu_to_au, sigfigs = 8)
                    if self.reducedMass == (None,): self.reducedMass = (reduced_mass,)
                    rounded_reducedMass = tuple(round(reduced_mass, sigfigs = 8) for reduced_mass in self.reducedMass)
                    assert reduced_mass in rounded_reducedMass, f"The reduced mass in the molscat output should be an element of {self}.reducedMass."
                    reduced_mass_index = rounded_reducedMass.index(reduced_mass)

                # determine which basis set is used in the output
                elif "in a total angular momentum basis" in line:
                    tcpld = True
                    fmfbasis = False
                    basis = ('L', 'F1', 'F2', 'F12', 'T', 'MT')
                    self.Qn = qn.Tcpld
                    self.diagonal = ('T', 'MT')

                    if self.basis == None: self.basis = basis
                    assert basis == self.basis, f"The basis set used in the molscat output should match {self}.basis."

                elif "ATOM A WITH S" in line:
                    S1 = int(line.split()[5].split(r'/')[0])
                    line = next(molscat_output)
                    I1 = int(line.split()[2].split(r'/')[0])

                elif "ATOM B WITH S" in line:
                    S2 = int(line.split()[5].split(r'/')[0])
                    line = next(molscat_output)
                    I2 = int(line.split()[2].split(r'/')[0])

                elif "L UP TO" in line:
                    L_max = 2*int(line.split()[3])

                elif "SPIN-SPIN TERM INCLUDED" in line:
                    spin_spin = True
                    if non_molscat_so_parameter is not None:
                        if self.spinOrbitParameter == (None,):
                            self.spinOrbitParameter = (non_molscat_so_parameter,)
                        
                        # since the scaling in molscat output has anyway 12 digits after the decimal point
                        rounded_spinOrbitParameter = tuple( round(so_param, decimals = 12) for so_param in self.spinOrbitParameter)
                        if round(non_molscat_so_parameter, decimals = 12) not in rounded_spinOrbitParameter:
                            raise ValueError(f"{non_molscat_so_parameter=} should be an element of {self.spinOrbitParameter=}")
                        
                        so_param_index = rounded_spinOrbitParameter.index(round(non_molscat_so_parameter, decimals = 12))
                    ## now, if non_molscat_so_parameter is not None, we are sure that it is an element of self.spinOrbitParameter
                    ## if non_molscat_so_parameter is None, we do nothing


                elif "SPIN-SPIN TERM OMITTED" in line:
                    spin_spin = False
                    so_param_index = 0
                    if non_molscat_so_parameter is not None:
                        raise ValueError(f"You defined the {non_molscat_so_parameter=}, but the spin-spin term is not included in this molscat calculation! For such calculations, only non_molscat_so_parameter = None is permitted.")
                    if self.spinOrbitParameter != (None,):
                        raise ValueError(f"The spin-orbit coupling parameter is defined in self.spinOrbitParameter, but the spin-spin term is not included in this molscat calculation!")
                    ## we don't want to fake data (on purpose or accidentaly) ;)

                elif "INTERACTION TYPE IS  ATOM - ATOM IN UNCOUPLED BASIS" in line:
                    tcpld = False
                    fmfbasis = True
                    # extra_operator_values = []
                    # raise NotImplementedError("Only the (L F1 F2 F12 T MT) basis can be used in the molscat outputs in the current implementation.")
                    basis = ('L', 'ML', 'F1', 'MF1', 'F2', 'MF2')
                    self.Qn = qn.LF1F2
                    if spin_spin == False: self.diagonal = ('ML')
                    if self.basis == None: self.basis = basis
                    assert basis == self.basis, f"The basis set used in the molscat output should match {self}.basis."
                
                elif "SHORT-RANGE POTENTIAL 1 SCALING FACTOR" in line:
                    # find values of short-range factors and C4
                    # the scaling in molscat output has anyway 12 digits after the decimal point
                    A_s = round(float(line.split()[9])*float(line.split()[11]), decimals = 12)      
                    if self.singletParameter == (None,): self.singletParameter = (A_s,)
                    rounded_singletParameter = tuple(round(singlet_parameter, decimals = 12) for singlet_parameter in self.singletParameter)
                    assert A_s in rounded_singletParameter, f"The singlet scaling parameter ({A_s=}) from the molscat output ({file_path=}) should be an element of {self}.singletParameter."
                    A_s_index = rounded_singletParameter.index(A_s)

                    for i in range(2):
                        line = next(molscat_output)
                        if "C 4 =" in line:
                            C4 = float(line.split()[3])
                    if self.C4 == (None,): self.C4 = (C4,)
                    assert C4 in self.C4, f"The value of C4 from the molscat output should be an element of {self}.C4."
                    C4_index = self.C4.index(C4)
                
                elif "SHORT-RANGE POTENTIAL 2 SCALING FACTOR" in line:
                    # the scaling in my molscat outputs has anyway 12 digits after the decimal point
                    A_t = round(float(line.split()[9])*float(line.split()[11]), decimals = 12)
                    
                    if self.tripletParameter == (None,): self.tripletParameter = (A_t,)
                    rounded_tripletParameter = tuple(round(triplet_parameter, decimals = 12) for triplet_parameter in self.tripletParameter)
                    assert A_t in rounded_tripletParameter, f"The triplet scaling parameter ({A_t=}) from the molscat output ({file_path=}) should be an element of {self}.tripletParameter."
                    A_t_index = rounded_tripletParameter.index(A_t)

                elif "SHORT-RANGE POTENTIAL 3 SCALING FACTOR" in line:
                    # the scaling in molscat output has anyway 12 digits after the decimal point
                    rkhs_ss_scaling = round(float(line.split()[9])*float(line.split()[11]), decimals = 12)
                    
                    # if non_molscat_so_parameter was None and the set of spin-orbit parameters
                    # was not defined while creating the SMatrixCollection object,
                    # we take the scaling of the spin-spin term in RKHS from the molscat output
                    if rkhs_ss_scaling != 1.0 and non_molscat_so_parameter is not None:
                        raise ValueError(f"The spin-spin term was scaled internally in molscat. You can't use the non_molscat_so_parameter in this case (I don't know how to do compare/multiply this parameters).")
                    if non_molscat_so_parameter is None:
                        if self.spinOrbitParameter == (None,) and spin_spin:
                            self.spinOrbitParameter = (rkhs_ss_scaling,)
                        rounded_spinOrbitParameter = tuple(round(so_param, decimals = 12) for so_param in self.spinOrbitParameter)
                        assert rkhs_ss_scaling in rounded_spinOrbitParameter, f"The spin-orbit scaling parameter {rkhs_ss_scaling=} from the molscat output ({file_path=}) should be an element of {self}.tripletParameter."
                        so_param_index = rounded_spinOrbitParameter.index(rkhs_ss_scaling)

                elif "INPUT ENERGY LIST IS" in line:
                    while "CALCULATIONS WILL BE PERFORMED FOR" not in line:
                        line = next(molscat_output)
                    line = next(molscat_output)
                    # create the list of energies from the output
                    energy_list = []
                    # append each energy value from the output to the list of energies
                    while line.strip():
                        # the energies in the molscat outputs are listed in G17.10 format here anyway
                        energy_list.append(round(float(line.split()[6]), decimals = 10))
                        line = next(molscat_output)
                    energy_tuple = tuple(energy_list)

                    if self.collisionEnergy == (None,): self.collisionEnergy = energy_tuple
                    rounded_collisionEnergy = tuple(round(energy, decimals = 10) for energy in self.collisionEnergy )
                    assert energy_tuple == rounded_collisionEnergy, f"The list of collision energies ({energy_list=}) from the molscat output ({file_path=}) should be equal to {self}.collisionEnergy."

                # elif "THESE ENERGY VALUES ARE RELATIVE TO THE REFERENCE ENERGY SPECIFIED BY MONOMER QUANTUM NUMBERS" in line:
                #     f1ref, mf1ref, f2ref, mf2ref = int(line.split()[14])/2, int(line.split()[15])/2, int(line.split()[16])/2, int(line.split()[17])/2 

                elif "EXTRA OPERATORS WILL BE USED TO RESOLVE ASYMPTOTIC DEGENERACIES" in line:
                    line = next(molscat_output)
                    line = next(molscat_output)
                    nextra = int(line.split()[2])
                    sums_of_MF_squares = {}
                    sums_of_MF = {}
                
                elif "*****************************  ANGULAR MOMENTUM JTOT  =" in line and tcpld:
                    T = int(line.split()[5])  
                    energy_counter = 0

                elif "*****************************  ANGULAR MOMENTUM JTOT  =" in line and fmfbasis:
                    MT = int(line.split()[5])
                    energy_counter = 0

                elif "MAGNETIC Z FIELD =" in line:
                    magnetic_field = float(line.split()[13])

                    if fmfbasis: thresholds = allThresholds(magnetic_field, 0, L_max, MT, I1, I2, identical = False)

                    if self.magneticField == (None,): self.magneticField = (magnetic_field,)
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
                
                elif "Eigenvalues of all operators for" in line and fmfbasis:
                    line = next(molscat_output)
                    while line.strip():
                        if nextra == 2:
                            sums_of_MF_squares[int(line.split()[1])] = float(line.split()[3])
                            sums_of_MF[int(line.split()[1])] = float(line.split()[4])
                        elif nextra == 1:
                            sums_of_MF_squares[int(line.split()[1])] = float(line.split()[3])
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
                
                elif "OPEN CHANNEL   WVEC (1/ANG.)    CHANNEL" in line and fmfbasis:
                    channels = {}
                    channel_in_indices = []
                    line = next(molscat_output)
                    while line.strip():
                        # get the open-channel index
                        open_channel_index = int(line.split()[0])
                        # convert the wavevector from 1/angstrom to 1/bohr
                        wavevector = float(line.split()[1])*bohrtoAngstrom
                        # convert the channel collision energy from hartrees to kelvins
                        channel_collision_energy = Hartree_to_K*(wavevector)**2/(2*reduced_mass)  
                        # get the index of the channel
                        channel_index = int(line.split()[2])
                        # append the index of the channel if matching the collision energy with the tolerance of 1e-6
                        if np.around(channel_collision_energy/self.collisionEnergy[energy_counter] - 1, decimals = 6) == 0: channel_in_indices.append(open_channel_index)
                        # get the doubled L quantum number
                        L = 2*int(line.split()[3])
                        # convert the channel's pair energy from cm-1 to kelvins
                        channel_pair_energy = float(line.split()[4]) / K_to_cm
                        # get the open channel's sum of MF, sum of MF squares, and ML
                        _sum_of_MF = sums_of_MF[channel_index]
                        _sum_of_MF_squares = sums_of_MF_squares[channel_index]
                        _ML = MT - _sum_of_MF
                        # get the possible MF1, MF2 values
                        _delta_square_root = np.sqrt(2*_sum_of_MF_squares - _sum_of_MF**2)
                        possible_MF_pairs = ( (_sum_of_MF - _delta_square_root)/2, (_sum_of_MF + _delta_square_root)/2 )
                        possible_MF_pairs = tuple( MF_pair for MF_pair in itertools.permutations(possible_MF_pairs)
                                              if MF_pair[0] in range(-(I1+S1), I1+S1+1, 2)
                                               if MF_pair[1] in range(-(I2+S2), I2+S2+1, 2) )
                        # get the possible channels (matching the pair energy and extra operator values)
                        #if MT == 5: print(thresholds)
                        possible_channels = tuple( qn.LF1F2(L, _ML, F1, MF_pair[0], F2, MF_pair[1])
                                                    for MF_pair in possible_MF_pairs
                                                        for F1 in range(abs(I1-S1), I1+S1+1, 2)
                                                            for F2 in range(abs(I2-S2), I2+S2+1, 2)
                                                                if MF_pair[0] in range(-F1, F1+1, 2)
                                                                 and MF_pair[1] in range(-F2, F2+1, 2)
                                                                  and np.around(channel_pair_energy/thresholds[qn.LF1F2(L, _ML, F1, MF_pair[0], F2, MF_pair[1])] - 1, decimals = 6) == 0)
                        if len(possible_channels) == 0 or len(possible_channels) > 1: raise ValueError(f"Couldn't resolve degeneracies: found {len(possible_channels)} channels matching the pair energy and extra operator values from the output.")
                        channels[open_channel_index] = possible_channels[0]
                        line = next(molscat_output)

                elif "ROW  COL       S**2                  PHASE/2PI" in line:
                    line = next(molscat_output)
                    matrix = {}
                    
                    while line.strip():
                        channel_out_index, channel_in_index = int(line.split()[0]), int(line.split()[1])
                        if channel_in_index in channel_in_indices:
                            matrix[(channels[channel_out_index], channels[channel_in_index])] = cmath.rect(np.sqrt(float(line.split()[2])), 2*np.pi*float(line.split()[3]))
                        line = next(molscat_output)
                    
                    S = SMatrix(basis = basis, diagonal = self.diagonal, C4 = self.C4[C4_index], singletParameter = self.singletParameter[A_s_index], tripletParameter = self.tripletParameter[A_t_index], spinOrbitParameter = self.spinOrbitParameter[so_param_index], reducedMass = self.reducedMass[reduced_mass_index], magneticField = self.magneticField[magnetic_field_index], collisionEnergy = self.collisionEnergy[energy_counter], matrix = matrix)
                    if (C4_index, A_s_index, A_t_index, so_param_index, reduced_mass_index, magnetic_field_index, energy_counter) in self.matrixCollection.keys():
                        self.matrixCollection[CollectionParametersIndices(C4_index, A_s_index, A_t_index, so_param_index, reduced_mass_index, magnetic_field_index, energy_counter)].update(S)
                    else:
                        self.matrixCollection[CollectionParametersIndices(C4_index, A_s_index, A_t_index, so_param_index, reduced_mass_index, magnetic_field_index, energy_counter)] = S

                    energy_counter = (energy_counter + 1) % len(self.collisionEnergy)
            
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
        # print(f'{qn_in=}, {qn_out=}', flush=True)
        return rate_coefficient_array

    def getMomentumTransferRateCoefficientVsL(self, qn_in: qn.LF1F2 | qn.LF12 | qn.Tcpld, unit = None, **kwargs) -> np.ndarray[Any, float]:
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
        rate_coefficient_array = np.array( [ self.matrixCollection[CollectionParametersIndices(*indices_combination)].getMomentumTransferRateCoefficientVsL(qn_in, unit = unit) for indices_combination in itertools.product( *param_indices ) ] )
        rate_coefficient_array = rate_coefficient_array.reshape( -1, *(len(index_tuple) for index_tuple in param_indices), rate_coefficient_array.shape[-1] )
        print(f'{rate_coefficient_array = }', flush = True)
        print(f'{rate_coefficient_array.shape = }', flush = True)
        return rate_coefficient_array


    def getMomentumTransferRateCoefficient(self, qn_in: qn.LF1F2 | qn.LF12 | qn.Tcpld, unit = None, **kwargs) -> np.ndarray[Any, float]:
        """Get the energy-dependent momentum-transfer rate coefficient for
          the given initial state as an array for the given parameter values,
          summed over all partial waves.

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


    def thermalAverage(self, array_to_average: np.ndarray[Any, float], distribution_array: np.ndarray[Any, float] = None) -> np.ndarray[Any, float]:
        """Thermally average an array of values.

        :param array_to_average: array of energy-depending values
          in the last axis.
        :param distribution_array: array providing
          the distribution factors in the integral.
        :return: array of the thermally averaged values,
          with the last axis contracted with respect to array_to_average.
        """


        if distribution_array is None:
            distribution_array = np.fromiter( n_root_iterator(temperature = 5e-4, E_min = min(self.collisionEnergy), E_max = max(self.collisionEnergy), N = len(self.collisionEnergy), n = 3), dtype = float )

        integrand = array_to_average * distribution_array
        integral = scipy.integrate.simpson( integrand )
        norm = scipy.integrate.simpson( distribution_array )

        averaged_array = integral / norm

        return averaged_array


    def getThermallyAveragedRate(self, qn_out: qn.LF1F2 | qn.LF12 | qn.Tcpld, qn_in: qn.LF1F2 | qn.LF12 | qn.Tcpld, distribution_array: np.ndarray[Any, float] = None, unit: str = None, **kwargs) -> np.ndarray[Any, float]:
        """Get the thermally averaged rate coefficient for the given final
          and initial state as an array for the given parameter values.

        :param qn_out: quantum numbers for the final state.
        :param qn_in: quantum numbers for the initial state.
        :param distribution_array: array providing
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
        averaged_rate_array = self.thermalAverage(rate_array_to_average, distribution_array)
        return averaged_rate_array

    def getThermallyAveragedMomentumTransferRateVsL(self, qn_in: qn.LF1F2 | qn.LF12 | qn.Tcpld, distribution_array: np.ndarray[Any, float] = None, unit: str = None, **kwargs) -> np.ndarray[Any, float]:
        """Get the thermally averaged momentum-transfer rate coefficient
          for the given final and initial state as an array
            for the given parameter values.

        :param qn_in: quantum numbers for the initial state.
        :param distribution_array: array providing
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

        rate_array_to_average = self.getMomentumTransferRateCoefficientVsL(qn_in, unit, **kwargs)
        averaged_rate_array = self.thermalAverage(rate_array_to_average, distribution_array)
        return averaged_rate_array


    def getThermallyAveragedMomentumTransferRate(self, qn_in: qn.LF1F2 | qn.LF12 | qn.Tcpld, distribution_array: np.ndarray[Any, float] = None, unit: str = None, **kwargs) -> np.ndarray[Any, float]:
        """Get the thermally averaged momentum-transfer rate coefficient
          for the given final and initial state as an array
            for the given parameter values, summed over all partial waves.

        :param qn_in: quantum numbers for the initial state.
        :param distribution_array: array providing
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
        averaged_rate_array = self.thermalAverage(rate_array_to_average, distribution_array)
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


    def getThermallyAveragedProbability(self, qn_out: qn.LF1F2 | qn.LF12 | qn.Tcpld, qn_in: qn.LF1F2 | qn.LF12 | qn.Tcpld, qn_momentum_transfer: qn.LF1F2 | qn.LF12 | qn.Tcpld = None, distribution_array: np.ndarray[Any, float] = None, **kwargs) -> np.ndarray[Any, float]:
        """Get the thermally averaged probability of collision calculated
          as the ratio of the energy-dependent rate coefficient for the
            transition from the given initial to the final state, and
              the momentum-transfer rate coeffient.

        :param qn_out: quantum numbers for the final state.
        :param qn_in: quantum numbers for the initial state.
        :param qn_momentum_transfer: quantum numbers for the initial state
          for calculating the momentum-transfer rate coefficient.
        :param distribution_array: array providing
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
        averaged_probability_array = self.thermalAverage(probability_array_to_average, distribution_array)
        return averaged_probability_array


    def getProbabilityFromThermalAverages(self, qn_out: qn.LF1F2 | qn.LF12 | qn.Tcpld, qn_in: qn.LF1F2 | qn.LF12 | qn.Tcpld, qn_momentum_transfer: qn.LF1F2 | qn.LF12 | qn.Tcpld = None, distribution_array: np.ndarray[Any, float] = None, **kwargs) -> np.ndarray[Any, float]:
        """Get the probability of collision calculated as the ratio of the
          thermally averaged rate coefficient for the transition from
            the given initial to the final state, and the momentum-transfer
              rate coefficient.

        :param qn_out: quantum numbers for the final state.
        :param qn_in: quantum numbers for the initial state.
        :param qn_momentum_transfer: quantum numbers for the initial state
          for calculating the momentum-transfer rate coefficient.
        :param distribution_array: array providing
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

        averaged_rate_array = self.getThermallyAveragedRate(qn_out, qn_in, distribution_array, **kwargs)
        averaged_momentum_transfer_rate_array = self.getThermallyAveragedMomentumTransferRate(qn_momentum_transfer, **kwargs)
        probability_from_averages_array = averaged_rate_array / averaged_momentum_transfer_rate_array
        return probability_from_averages_array



def main():
    print("Hello, I have nothing to do :)")

if __name__ == '__main__':
    main()