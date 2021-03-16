
import numpy as np
from math import factorial
import itertools
import scipy
import scipy.special as sp

dbl_quad = scipy.integrate.dblquad

def Saxon_potential(r, V0, a, A):
    """Woods-Saxon Potential:
    V0: Potential depth ~ 50
    a: Nuclear surface thickness ~ 0.524
    A: Mass number
    R = 1.25*A**1/3
    """
    rR = r-1.25*A**(1/3)
    return -V0 / (1 + np.exp(rR/a))

class Harmonic:
    def basis(n):
        """Creates 1D hamonic oscillator basis: Hermite
        where m*w/hbar coefficient = 1
        """
        
        normer = 1/np.sqrt(2**n * factorial(n) * np.sqrt(np.pi))
        wave_rad = lambda x: normer * np.exp(-x**2 / 2) * sp.eval_hermite(n, x)
        return np.vectorize(wave_rad)
    
    def energies(n):
        return n + 1/2

def create_slaters(states_num, particles):
    """Creates possible slaters that add to given M value"""
    states = np.arange(1, states_num+1)
    slaters_pick = [slate for slate in itertools.combinations(states, particles)]
    return np.array(slaters_pick)

def slater_energy(slaters_m, energies):
    """
     Function that calculates th digaonal of the Hamiltonian
     by summing the single-particle energies of the Slater determinants"""
    
    return np.sum(energies[slaters_m-1], axis=1)

def interact(x):
    '''2-body Interaction function''' 
    return np.exp(-x**2)

def twobody_integral(i, j, k, l, basis):
    '''Calculates <ij|V|kl> with quantum number n of each state'''
    inter = lambda x1, x2: basis[i](x1)*basis[j](x2)*interact(x2-x1)*basis[k](x1)*basis[l](x2)
    return dbl_quad(inter,0,4,0,4)[0]

def integrals_n(n, basis):
    integrals = dict()
    for config in itertools.combinations_with_replacement(n, 4):
        integrals[config.tobytes()] = twobody_integral(*config, basis)
    return integrals

def create_matrix_elements(matrix_combs, basis, n):
    '''Finds All matrix elements for each possible ijkl combination'''
    
    integrals = integrals_n(n, basis)
    
    matrix_elem = []
    for comb in matrix_combs:
        inder = np.sort(n[comb-1]).tobytes()
        matrix_elem.append(integrals[inder])
    
    return matrix_elem

def create_matrix_combs(n):
    combs = []
    for comb1 in itertools.combinations(n, 2):
        for comb2 in itertools.combinations(n, 2):
            combs.append(comb1 + comb2)
    
create_matrix_combs(range(1, 5))