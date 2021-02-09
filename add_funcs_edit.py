# Main Functions for shell code
# EDIT: NEW ADD_FUNCS CODE - Adam Asaad

import itertools
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
from math import factorial
from sympy.combinatorics.permutations import Permutation


def permutator(array):
    """Finds signature of permutation and returns sorted array """
    
    # Check for duplicates + check if empty
    if len(array) != len(set(array)) or len(array) == 0:
        return 0, []
    
    # Check parity
    arger = np.argsort(array)
    sign = Permutation(arger).signature()
    return sign, tuple(np.sort(array))
    
def showlevels(energies):
    """Plots the given eigen-energies"""    

    for e in energies:
        plt.plot((1.0,1.5),(e,e),'k-') #draws a line at energy e
    plt.plot((0.5,2.0),(0,0),'r--')
    plt.ylabel('Energy')
    plt.title('Energy levels')
    plt.xlim([0,2.5])
    plt.tick_params(       # Remove x-tics:
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off') # labels along the bottom edge are off
    plt.show()
    return
    

def create_basis(k, l, m):
    """Creates 3D hamonic oscillator basis vectorized function: Radials + Spherical
    where v coefficient = 1  
    """
    
    N_kl = np.sqrt(np.sqrt(2/np.pi) * (2**(k+2*l+3) * factorial(k)) / sp.factorial2(2*k+2*l+1))   
    wave_rad0 = lambda r: N_kl * r**l * np.exp(-1*r**2)
    wave_rad1 = lambda r: sp.eval_genlaguerre(k, l+1/2, 2*r**2)
    
    wave = lambda r, thet, phi: wave_rad0(r)*wave_rad1(r)*sp.sph_harm(m, l, thet, phi)
    return np.vectorize(wave)

def create_radial(k, l):
    """Creates 3D hamonic oscillator basis: Only Radials
    where v coefficient = 1
    """
    
    N_kl = np.sqrt(np.sqrt(2/np.pi) * (2**(k+2*l+3) * factorial(k)) / sp.factorial2(2*k+2*l+1))   
    wave_rad0 = lambda r: N_kl * r**l * np.exp(-1*r**2)
    wave_rad1 = lambda r: sp.eval_genlaguerre(k, l+1/2, 2*r**2)
    
    wave = lambda r: wave_rad0(r)*wave_rad1(r)
    return np.vectorize(wave)

#==============================================================================================
# Routine that picks only those slater
# determinants of the pairing problem
#
# E.g. with input:
# myslaters=[[1,2,3,4],[1,2,4,5],[3,4,5,8],[1,2,7,8],[3,5,6,7], [3,4,5,6], [3,4,7,8], [1,6,7,8]]
# you will get
#
# [[1, 2, 3, 4], [1, 2, 7, 8], [3, 4, 5, 6], [3, 4, 7, 8]]
#==============================================================================================
def pickpairs(slaters):
    suitableslaters = []
    for slate in slaters:
        slate = np.array(slate)
        if slate[0::2] % 2 != 0 and slate[1::2] % 2 == 0 and slate[::2] == slate[1::2] + 1:
            suitableslaters.append(slate)
    return suitableslaters

#==============================================================================================
# Routine that picks only those slater
# determinants of the pairing problem
#
# E.g. with input:
#
# n=np.array([0,0,1,1,2,2,3,3])
# l=np.array([0,0,0,0,0,0,0,0])
# j=np.array([1,1,1,1,1,1,1,1])
# myslaters=[[1,2,3,4],[1,2,4,5],[3,4,5,8],[1,2,7,8],[3,5,6,7], [3,4,5,6], [3,4,7,8], [1,6,7,8]]
#
# you will get
# [[1, 2, 3, 4], [1, 2, 7, 8], [3, 4, 5, 6], [3, 4, 7, 8]]
#==============================================================================================

def pickpairsnljm2(slaters, n, l, j):
   suitableslaters = []
   for ind, slate in enumerate(slaters):     
      for i in range(0, len(slate), 2):
         if n[slate[i]-1] != n[slate[i+1]-1] \
         or l[slate[i]-1] != l[slate[i+1]-1] \
         or j[slate[i]-1] != j[slate[i+1]-1]:
             break
      else:
          suitableslaters.append(slate)
          continue
      break
  
   return suitableslaters

#==============================================================================================
# Functions that returns a set of set of possible 
# Slater determinant for set of states s and m number of particles
# Picks the slaterdeterminants which have given M
# the sum is equals to given M, slater determinant is accepted
#
# For example, with given input
# myslaters=[[1,2,3,4],[1,3,5,7],[3,5,6,7],[1,3,7,8],[2,4,6,8], [3,4,5,6], [4,5,6,7], [1,6,7,8]]
# m=np.array([-1,1,-1,1,-1,1,-1,1])
# M=0
# you get
# [[1, 2, 3, 4], [3, 4, 5, 6], [4, 5, 6, 7], [1, 6, 7, 8]]
#==============================================================================================

# returns picked slaters and l
def create_slaters(states_num, particles, m_j, M2):
    """Creates possible slaters that add to given M value"""
    states = np.arange(1, states_num+1)
    slaters_pick = []
   
    for slate in itertools.combinations(states, particles):
       sum_m = sum(m_j[i-1] for i in slate)
       if sum_m == M2:
           slaters_pick.append(slate)     
    return slaters_pick


def two_body(p, q, r, s, alpha, sd_pairs):
    """
     Function that takes two-body operator indices p,q,r,s and
     index alpha for the ket Slater determinant and returns the index beta
     of the new Slater determinant and the phase"""
    
    slate = sd_pairs[alpha]
    if (r in slate) and (s in slate) and (p != q) and (r != s):  
        if (p == r) and (q == s):
            (sign, slater) = permutator(slate)
        else:
            # Replace the r- and s-values in alpha with p and q.
            slater0 = np.copy(slate)
            slater0[slate.index(s)] = q
            slater0[slate.index(r)] = p
            (sign, slater) = permutator(slater0)

        try:
            beta = sd_pairs.index(slater)
        except:
            beta, sign = (0, 0)
        return (beta, sign)
    
    # New slater determinant is zero (set phase to zero and index to N_slater+10)
    beta = len(sd_pairs) + 10
    return (beta, 0)


def diag_element(slater, energies):
    """
     Function that calculates a diagonal element in the Hamiltonian
     by summing the single-particle energies of the Slater determinant"""
    
    return sum(energies[i-1] for i in slater)
