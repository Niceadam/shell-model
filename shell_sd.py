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
    N_slater = len(slaters_m)
    
    # Possible n values
    poss_n = list(set(n))
                  
    # Create basis orbitals - (n=0, n=1)
    basis = [create_radial_1D(n_i) for n_i in poss_n]
    matrix_elem = create_matrix_elements(poss_n, basis)
    
    # Build Hamiltonian matrix
    H = np.zeros((N_slater, N_slater))
    for i in tqdm(range(N_slater)):
        slate = slaters_m[i]
        
        # Add single-particle energies to diagonal elements
        H[i,i] += slater_energy(slate, sp_energies)
        
        # Add matrix elements
        for i, j, k, l in zip(im, jm, km, lm):
            (ind, sign) = two_body(i, j, k, l, slate, slaters_m)

            if sign != 0:
                # Get matrix element for given config
                config = np.sort(n[[i-1, j-1, k-1, l-1]])
                elem = sign * matrix_elem[tuple(config)]
                if k == i:
                    H[i,i] += elem
                else:
                    H[ind,i] += elem
                    H[i,ind] += elem
    
    # Diagonalize the Hamiltonian matrix
    eigs, vecs = linalg.eig(H)
    vecs = vecs.T
    
    return eigs, vecs, slaters_m, basis
    
    
def plot_radials(weights, slaters_m, basis, rlim=[0.01, 3], labeler='Ground'):
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
    
#################################
# Main Function

N_particles, M_val = 8, 8
eigs, vecs, slaters_m, basis = main_eigen(N_particles, M_val)

plot_radials(vecs[0], slaters_m, basis, labeler='Ground')
plt.xlabel('Radius (r)'); plt.ylabel('Valence Density')
plt.grid()
plt.legend()