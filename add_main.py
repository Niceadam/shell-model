
import numpy as np
from math import factorial, sqrt, exp
import matplotlib.pyplot as plt
from itertools import combinations, combinations_with_replacement
from sympy.utilities.iterables import multiset_permutations
from scipy.special import eval_hermite
from scipy.integrate import dblquad, quad
from numpy import pi, inf
from tqdm import tqdm
from multiprocessing import Pool

tol = 1e-4 # tolerance for integration
bnd = 12 # Integration bound

########### Harmonic Basis
m = 2
w = 1
hbar = 1
mwh = m*w/hbar

def harmonic_basis(n):
    """Creates 1D hamonic oscillator basis: Hermite
    """
    normer = 2**(-n/2) / sqrt(factorial(n)) * pow(mwh / pi, 1/4)
    wave_rad = lambda x: normer * exp(-mwh/2 * x**2) * eval_hermite(n, sqrt(mwh)*x)
    return np.vectorize(wave_rad)

########### Potential

@np.vectorize
def harmonic_potential(x, m, w):
    return 0.5*m*w**2 * x**2

@np.vectorize
def woods_saxon_potential(x, V0, a, A):
    inper = (abs(x) - 1.25*pow(A, 1/3)) / a
    
    if inper > 80: 
        return 0 
    else:
        return -V0 / (1+exp(inper))
    
potential = lambda x: harmonic_potential(x, 2, 1)
#potential = lambda x: woods_saxon_potential(x, 10, 0.2, 20)+10
#potential = lambda x: square_well(x, 4)

# x = np.linspace(-8, 8, 100)
# plt.plot(x, potential(x))
# plt.plot(x, harmonic_potential(x, 1, 1))


########### Slater creation
def create_slaters(states_num, particles):
    """Creates possible slaters that add to given M value"""
    
    limer = "1"*particles + "0"*(states_num-particles)
    nums = [''.join(x) for x in multiset_permutations(limer)]
    return nums

########### 2-body elements

def twobody_element(i, j, k, l, b):
    """<ij|kl> - <ij|lk> element"""
    
    inter = lambda x1, x2: b[i](x1)*b[j](x2) *exp(-2*(x2-x1)**2)* (b[k](x1)*b[l](x2) - b[l](x1)*b[k](x2))
    return dblquad(inter, -bnd, bnd, -bnd, bnd, epsabs=tol)[0]

def create_elements2(N_sp, basis):
    elems = dict()
    combers = list(combinations(range(N_sp), 2))
    
    global itercomb
    def itercomb(ind, i):
        elems_current = dict()
        for j in combers[ind:]:
            if sum(i+j) & 1 == 0 and (N_sp-1 in i+j):
                elems_current[i+j] = twobody_element(*i, *j, basis)
        return elems_current
    
    with Pool() as pool:
        results = pool.starmap_async(itercomb, enumerate(combers)).get()
    
    for i in results: elems.update(i)
    return elems

########### 1-body elements
def onebody_element(i, j, b):
    """<i|t + v|j> element"""
    
    hterm = lambda x: b[i](x)*b[j](x) * (-0.5 * mwh*(mwh*x**2-2*j-1) + potential(x))
    return quad(hterm, -bnd, bnd, epsabs=tol)[0]

def create_elements1(N_sp, basis):
    elems = dict()
    combers = list(combinations_with_replacement(range(N_sp), 2))
    for i in tqdm(combers):
        if sum(i) & 1 == 0:
            elems[i] = onebody_element(*i, basis)
    return elems

def slater_energ(elements0, slaters_m):
    new = []
    for slater in slaters_m:
        sumer = sum(elements0[i] for i, biner in enumerate(slater) if biner == '1')
        new.append(sumer)
    
    return np.array(new)