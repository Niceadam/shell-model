# Nuclear TALENT 2017
# Tiia, Shane, Martin, Ovidiu
# This code reads the single-particle states and effective interactions 
# for N particles and total angular momentum M and plots the energy eigenvalues

# EDIT: NEW SHELL CODE - Adam Asaad

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg
from timeit import default_timer as timer
from tqdm import tqdm
from add_funcs import *


# Load single-particle data from file 'sdshellint.dat' ordered as (index, n, l, 2j, 2m_j, energies)
sp_data = np.loadtxt('input_data/spdata_sd.dat',skiprows=2,usecols=[0,1,2,3,4,5])

# Number of single-particle states + Data Unpacking
N_sp = len(sp_data)
n = sp_data[:,1].astype(int)
l = sp_data[:,2].astype(int)
j = sp_data[:,3].astype(int)
m_j = sp_data[:,4].astype(int)
sp_energies = sp_data[:,5]

# Load effective interactions from file 'shellint_sd.dat' ordered as (i,j,k,l,<ij|V|kl>
int_data = np.loadtxt('input_data/shellint_sd.dat',skiprows=2,usecols=[0,1,2,3,4])

im = int_data[:,0].astype(int)
jm = int_data[:,1].astype(int)
km = int_data[:,2].astype(int)
lm = int_data[:,3].astype(int)
matrix_combs = np.array(list(zip(im, jm, km, lm)))

def main_eigen(N_particles, M_val):    
    """
    Returns Eigen-everything from CI for N particles and total angular momentum M"""
    
    # N_paticles must be less than or equal to number of single-particle states)
    if N_particles > N_sp:
        raise "Number of particles must be less than number of single-particle states!"
    
    # total 2*M of the system must be same parity as number of particles
    if (N_particles & 1) != (M_val & 1):
        raise 'Must match parity!'
    
    
    # Number of matrix elements of effective interaction + scaling factor
    matrix_elements = int_data[:,4]
    scaling_factor = (18.0 / (16.0+N_particles))**0.3
    matrix_elements = scaling_factor * matrix_elements
    
    # Create possible Slater determinants with given 2*M
    slaters_m = create_slaters(N_sp, N_particles, m_j, M_val)
    test_m = [slaters_m.index(tuple(reversed(i))) for i in sorted([slate[::-1] for slate in slaters_m])]
    slaters_m = [slaters_m[i] for i in test_m]
    
    N_slater = len(slaters_m)
    
    # Possible n values
    poss_n = list(set(n))
                  
    # Create basis orbitals - (n=0, n=1)
    basis = [create_radial_1D(n_i) for n_i in poss_n]
    
    # Create matrix elements
    matrix_elem = create_matrix_elements(matrix_combs, basis, n)
    
    # Build Hamiltonian matrix
    # Add single-particle energies to diagonal elements
    slater_energies = slater_energy(slaters_m, sp_energies)
    H = np.diag(slater_energies)
    
    for i, slate in tqdm(enumerate(slaters_m)):
        # Add matrix elements: comb = (i, j, k, l)
        for a, comb in enumerate(matrix_combs):
            (ind, sign) = two_body(*comb, slate, slaters_m)
            
            if sign != 0:
                elem = sign * matrix_elem[tuple(comb)]
                #elem = sign * matrix_elements[a]
                if ind == i:
                    H[i,i] += elem
                else:
                    H[ind,i] += elem
                    H[i,ind] += elem
    
    # Diagonalize the Hamiltonian matrixs
    eigs, vecs = linalg.eig(H)
    vecs = vecs.T
    order = np.argsort(eigs)
    eigs, vecs = eigs[order], vecs[order]    
    
    return eigs, vecs, slaters_m, basis
        
#################################
# Main Function

N_particles, M_val = 4, 4
eigs, vecs, slaters_m, basis = main_eigen(N_particles, M_val)


plt.plot(vecs[0]**2, label='Ground')
plt.plot(vecs[1]**2, label='1st Excited')
plt.plot(vecs[2]**2, label='2nd Excited')
plt.legend()
plt.figure()

# plot_radials(vecs[0], slaters_m, basis, n, labeler='Ground')
# plt.xlabel('Radius (r)'); plt.ylabel('Valence Density')
# plt.grid()
# plt.legend()
