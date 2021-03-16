
import numpy as np
from math import factorial
import itertools
import scipy
import scipy.special as sp
import scipy.integrate

dbl_quad = scipy.integrate.dblquad

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
    
    limit = int("0b"+"1"*states_num, 2)
    
    nums = [np.binary_repr(x, states_num) for x in range(limit+1)]
    nums = list(filter(lambda x: x.count('1') == particles, nums))
    return nums


def slater_energy(slaters_m, energies):
    slater_energies = []
    for slate in slaters_m:
        slater_energies.append(sum(energies[[x == '1' for x in slate]]))
    return np.array(slater_energies)

def interact(x):
    '''2-body Interaction function''' 
    return np.exp(-x**2)

def matrix_element(comber, basis):
    '''Calculates <ij|V|kl> with quantum number n of each state'''
    
    i, j, k, l = 0, 0, 0, 0
    inter = lambda x1, x2: basis[i](x1)*basis[j](x2)*interact(x2-x1)*basis[k](x1)*basis[l](x2)
    return dbl_quad(inter, 0,4,0,4)[0]
