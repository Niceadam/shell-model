
import numpy as np
from math import factorial
import itertools
import scipy
import scipy.special as sp
import scipy.integrate
from sympy.utilities.iterables import multiset_permutations

dbl_quad = scipy.integrate.dblquad

class Harmonic:
    def basis(n):
        """Creates 1D hamonic oscillator basis: Hermite
        where m*w/hbar = cc
        """
        mw = 1
        
        normer = 1/np.sqrt(2**n * factorial(n)) * pow(mw/np.pi, 1/4)
        wave_rad = lambda x: normer * np.exp(-mw*x**2 / 2) * sp.eval_hermite(n,np.sqrt(mw)*x)
        return np.vectorize(wave_rad)
    
    def energies(n):
        return n + 1/2

def create_slaters(states_num, particles):
    """Creates possible slaters that add to given M value"""
    
    limer = "1"*particles + "0"*(states_num-particles)
    nums = [''.join(x) for x in multiset_permutations(limer)]
    return nums

def slater_energy(slaters_m, energies):
    slater_energies = [sum(energies[[x == '1' for x in slate]]) for slate in slaters_m]
    return np.array(slater_energies)

def interact(x):
    '''2-body Interaction function''' 
    return np.exp(-x**2)

def integraler(i, j, k, l, b):
    inter = lambda x1, x2: b[i](x1)*b[j](x2)*interact(x2-x1)*b[k](x1)*b[l](x2)
    return dbl_quad(inter, 0,4,0,4)[0]

def matrix_element2(comber, basis, N_sp):
    return 0

def matrix_element1(comber, basis, N_sp):
    return 0

def matrix_element0(comber, basis, N_sp):
    return 0