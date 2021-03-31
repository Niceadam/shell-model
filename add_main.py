
import numpy as np
from math import factorial
from itertools import combinations, product
from sympy.utilities.iterables import multiset_permutations
import scipy.special as sp
from scipy.integrate import dblquad, quad
from scipy.ndimage.filters import laplace
from numpy import exp, pi, sqrt
from tqdm import tqdm


############### Harmonic Basis

lener = 5 # integration bound
tol = 1e-3 # tolerance for integration

########### Harmonic Basis
m = 1
w = 1
hbar = 1
mwh = m*w/hbar

def harmonic_basis(n):
    """Creates 1D hamonic oscillator basis: Hermite
    """
    
    normer = 1/sqrt(2**n * factorial(n)) * pow(mwh / pi, 1/4)
    wave_rad = lambda x: normer * exp(-mwh/2 * x**2) * sp.eval_hermite(n,sqrt(mwh)*x)
    return np.vectorize(wave_rad)

########### Potential

def harmonic_potential(x, m, w):
    return 0.5*m*w*x**2

def woods_saxon_potential(x, V0, a, A):
    inper = (abs(x) - 1.25*pow(A, 1/3)) / a
    return -V0 / (1+exp(inper))
    
#potential = lambda x: harmonic_potential(x, 2, 1)
potential = lambda x: woods_saxon_potential(x, 1, 0.5, 16)
#potential = lambda x: np.sin(x)

########### Slater creation
def create_slaters(states_num, particles):
    """Creates possible slaters that add to given M value"""
    
    limer = "1"*particles + "0"*(states_num-particles)
    nums = [''.join(x) for x in multiset_permutations(limer)]
    return nums

########### 2-body elements

def twobody_element(i, j, k, l, b):
    """<ij|kl> - <ij|lk> element"""
    inter = lambda x1, x2: b[i](x1)*b[j](x2) *exp(-(x2-x1)**2)* (b[k](x1)*b[l](x2) - b[l](x1)*b[k](x2))
    return dblquad(inter, -lener, lener, -lener, lener, epsabs=tol)[0]

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
    
    hterm = lambda x: b[i](x)*b[j](x) * (-0.5 * mwh**2 * (mwh*x**2-2*j-1) + potential(x))
    return quad(hterm, -lener-1, lener+1, epsabs=tol)[0]


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