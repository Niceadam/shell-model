# Main Functions for shell code
# EDIT: NEW ADD_FUNCS CODE - Adam Asaad

import itertools
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
import scipy.integrate
from math import factorial
from sympy.combinatorics.permutations import Permutation
from read_lpt import read_lpt, read_lpt_exp
from tqdm import tqdm

dbl_quad = scipy.integrate.dblquad

def permutator(array):
    """Finds signature of permutation and returns sorted array """
    
    # Check for duplicates + check if empty
    if len(array) != len(set(array)) or len(array) == 0:
        return 0, []
    
    # Check parity
    arger = np.argsort(array)
    sign = Permutation(arger).signature()
    return sign, np.sort(array)
    
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
    

def create_basis_3D(k, l, m):
    """Creates 3D hamonic oscillator basis vectorized function: Radials + Spherical
    where v coefficient = 1  
    """
    
    N_kl = np.sqrt(np.sqrt(2/np.pi) * (2**(k+2*l+3) * factorial(k)) / sp.factorial2(2*k+2*l+1))   
    wave_rad0 = lambda r: N_kl * r**l * np.exp(-1*r**2)
    wave_rad1 = lambda r: sp.eval_genlaguerre(k, l+1/2, 2*r**2)
    
    wave = lambda r, thet, phi: wave_rad0(r)*wave_rad1(r)*sp.sph_harm(m, l, thet, phi)
    return np.vectorize(wave)

def create_radial_3D(k, l):
    """Creates 3D hamonic oscillator basis: Only Laguerre Radials
    where v coefficient = 1
    """
    
    N_kl = np.sqrt(np.sqrt(2/np.pi) * (2**(k+2*l+3) * factorial(k)) / sp.factorial2(2*k+2*l+1))   
    wave_rad0 = lambda r: N_kl * r**l * np.exp(-1*r**2)
    wave_rad1 = lambda r: sp.eval_genlaguerre(k, l+1/2, 2*r**2)
    
    wave = lambda r: wave_rad0(r)*wave_rad1(r)
    return np.vectorize(wave)

def create_radial_1D(n):
    """Creates 1D hamonic oscillator basis: Hermite
    where m*w/hbar coefficient = 1
    """
    
    const = 1/np.sqrt(2**n * factorial(n) * np.sqrt(np.pi))
    wave_rad = lambda x: const * np.exp(-x**2 / 2) * sp.eval_hermite(n, x)
    return np.vectorize(wave_rad)

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
    return np.array(slaters_pick)


def two_body(p, q, r, s, slate, slaters_m):
    """
     Function that takes two-body operator indices <pq|rs> and
     slater and returns the index where to insert matrix element and the phase"""
    
    if (r in slate) and (s in slate):       
        slater = np.copy(slate)
        
        # a_p+ a_q+ a_r a_s|p>
        slater[slate == r] = p
        slater[slate == s] = q
        
        (sign, slater) = permutator(slater)    
        if slater == []:
            return 0, 0
        else:
            ind = (slaters_m == slater).all(axis=1).nonzero()[0][0]
            return ind, sign
    
    # rs not in slater, so a_r,a_s|p> = 0
    else:
        return 0, 0
    

def interact(x):
    '''2-body Interaction function''' 
    return np.exp(-x**2)

def twobody_integral(i, j, k, l, basis):
    '''Calculates <ij|V|kl> with quantum number n of each state'''
    
    inter = lambda x1, x2: basis[i](x1)*basis[j](x2)*interact(x2-x1)*basis[k](x1)*basis[l](x2)
    return dbl_quad(inter, 0,4,0,4)[0]

def integrals_n(n, basis):
    
    integrals = dict()
    for config in itertools.combinations_with_replacement(set(n), 4):
        integrals[config] = twobody_integral(*config, basis)
    return integrals

def create_matrix_elements(matrix_combs, basis, n):
    '''Finds All matrix elements for each possible ijkl combination'''
    
    integrals = integrals_n(n, basis)
    
    matrix_elem = []
    for comb in tqdm(matrix_combs):
        inder = tuple(np.sort(n[comb-1]))
        matrix_elem.append(integrals[inder])
    
    return matrix_elem
        
def slater_energy(slaters_m, energies):
    """
     Function that calculates th digaonal of the Hamiltonian
     by summing the single-particle energies of the Slater determinants"""
    
    return np.sum(energies[slaters_m-1], axis=1)

def plot_radials(weights, slaters_m, basis, n, rlim=[0.01, 3], labeler='Ground'):
    """Given possible Slaters + CI weights for each slater = Plot Radial Denstiy function
    
    Only 2 Radial Functions: 
        1d: n=0 l=2
        2s: n=1 l=0
    """
    
    # Setup grid
    resol = 300 # resolution for the arrays used for plotting
    r = np.linspace(rlim[0], rlim[1], resol)
    
    # Evaluate orbital basis radials for each possible state
    basis = [base(r) for base in basis]
    
    # Final density = sum(weight[i]**2 * slater_dens[i])
    radial_dens = np.zeros(resol)
    for i, slate in enumerate(slaters_m):
        
        # Calculate Slater Density. You can't plot Slaters directly obviously!!
        slate_den = sum(basis[n[k-1]]**2 for k in slate)
                        
        # Add Slate density to Global Density with ground weight |c_i|^2
        radial_dens += weights[i]**2 * slate_den

    plt.plot(r, radial_dens, label=labeler)

def plot_energies(eigs, N_particles):
    """Gives the energy plot for given Eigen-energies"""
    
    # Include only levels below mev_limit MeV
    mev_lim = 6
    min_energy = eigs.min()
    E = eigs - min_energy
    E = E[E<mev_lim]

    
    if (N_particles % 2) == 0:
       # Compare with NushellX levels
       E_nu = read_lpt(N_particles)
       E_nu = E_nu[E_nu<mev_lim]
    
       E_exp = read_lpt_exp(N_particles)
       E_exp = E_exp[E_exp<mev_lim]
    
       for e in E_nu:
           plt.plot([1.5,2.5],[e,e],'r')
       for e in E_exp:
           plt.plot([-1.5,-0.5],[e,e],'k')
    
    for e in E:
    	plt.plot([0,1],[e,e],'b')
    
    plt.ylabel('Energy (MeV)', fontsize=20)
    plt.xticks([-1,0.5,2.0],['Exp','Calc','NushellX'])
    plt.ylim(-0.25, mev_lim)
    plt.title(r'$^{'+str(16+N_particles)+'}$'+'O', fontsize=20)
    plt.grid()
