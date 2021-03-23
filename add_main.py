
import numpy as np
from math import factorial
from itertools import combinations, product
from sympy.utilities.iterables import multiset_permutations
import scipy.special as sp
from scipy.integrate import dblquad, quad
from tqdm import tqdm


############### Harmonic Basis

# Parameters
mw = 1
lener = 4 # integration bound
tol = 1e-3 # tolerance for integration

@np.vectorize
def harmonic_potential(x):
    return 0.5*mw*x**2
        
def harmonic_basis(n):
    """Creates 1D hamonic oscillator basis: Hermite
    """
    
    normer = 1/np.sqrt(2**n * factorial(n)) * pow(mw/np.pi, 1/4)
    wave_rad = lambda x: normer * np.exp(-mw*x**2 / 2) * sp.eval_hermite(n,np.sqrt(mw)*x)
    return np.vectorize(wave_rad)

########### Slater creation
def create_slaters(states_num, particles):
    """Creates possible slaters that add to given M value"""
    
    limer = "1"*particles + "0"*(states_num-particles)
    nums = [''.join(x) for x in multiset_permutations(limer)]
    return nums

########### 2-body elements

def twobody_element(i, j, k, l, b):
    """<ij|kl> - <ij|lk> element"""
    inter = lambda x1, x2: b[i](x1)*b[j](x2) *np.exp(-(x2-x1)**2)* (b[k](x1)*b[l](x2) - b[l](x1)*b[k](x2))
    return dblquad(inter, 0,lener,0,lener, epsabs=tol)[0]

def create_elements2(N_sp, basis):
    elems = dict()
    combers = list(combinations(range(N_sp), 2))
    for ind, i in enumerate(tqdm(combers)):
        for j in combers[ind:]:
            elems[i+j] = twobody_element(*i, *j, basis)
    return elems

########### 1-body elements
def onebody_element(i, j, b):
    """<i|t + v|j> element"""
    
    hterm = lambda x: b[i](x) * b[j](x) * (0.5*(2*j + 1 - x**2) + harmonic_potential(x))
    return quad(hterm, 0,lener, epsabs=tol)[0]

def create_elements1(N_sp, basis):
    elems = dict()
    for i in tqdm(product(range(N_sp), repeat=2)):
        elems[i] = onebody_element(*i, basis)
    return elems

def slater_energ(elements0, slaters_m):
    new = []
    for slater in slaters_m:
        booler = [bool(int(x)) for x in slater]
        sumer = sum(elements0[booler])
        new.append(sumer)
    
    return np.array(new)