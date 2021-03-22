# Nuclear TALENT 2017
# Tiia, Shane, Martin, Ovidiu
# This code reads the single-particle states and effective interactions 
# for N particles and total angular momentum M and plots the energy eigenvalues

# EDIT: NEW SHELL CODE - Adam Asaad

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mat
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

def main_eigen(N_particles, M_val, data=True):    
    """
    Returns Eigen-everything from CI for N particles and total angular momentum M"""
    
    # N_paticles must be less than or equal to number of single-particle states)
    if N_particles > N_sp:
        raise "Number of particles must be less than number of single-particle states!"
    
    # total 2*M of the system must be same parity as number of particles
    if (N_particles & 1) != (M_val & 1):
        raise 'Must match parity!'
    
    
    # Number of matrix elements of effective interaction + scaling factor
    scaling_factor = (18.0 / (16.0+N_particles))**0.3
    matrix_elements = scaling_factor * int_data[:,4]
    
    # Create possible Slater determinants with given 2*M
    slaters_m = create_slaters(N_sp, N_particles, m_j, M_val)
    slater_energies = slater_energy(slaters_m, sp_energies)
    
    sorter = np.argsort(slater_energies)
    slater_energies = slater_energies[sorter]
    slaters_m = slaters_m[sorter]
    
    N_slater = len(slaters_m)
                  
    # Create basis orbitals - (n=0, n=1)
    basis = [create_radial_1D(n_i) for n_i in set(n)]
    
    # Create matrix elements
    if data == False:
        matrix_elements = create_matrix_elements(matrix_combs, basis, n)
    
    # Build Hamiltonian matrix
    # Add single-particle energies to diagonal elements
    H = np.diag(slater_energies)
    
    for i, slate in tqdm(enumerate(slaters_m)):
        # Add matrix elements: comb = <ij|kl>
        for a, comb in enumerate(matrix_combs):
            (ind, sign) = two_body(*comb, slate, slaters_m)
            
            if sign != 0:
                elem = sign * matrix_elements[a]
                    
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

N_particles, M_val = 5, 7
eigs, vecs, slaters_m, basis = main_eigen(N_particles, M_val, data=True)
eigs2, vecs2, slaters_m, basis = main_eigen(N_particles, M_val, data=False)

vecs = np.square(vecs)
vecs2 = np.square(vecs2)
lener = np.arange(len(slaters_m))

colors = []
for i in slaters_m:
    inter = sum([1 for j,k in zip(slaters_m[0], i) if j != k])
    colors.append(inter)
cm = mat.cm.get_cmap('jet')
normer = mat.colors.Normalize()
color = cm(normer(colors))

plt.bar(lener, vecs[0], alpha=0.7, color=color)
plt.xlabel('determinant number')
plt.ylabel('Magnitude of ground-state coefficient')
plt.title('N={}, 2M={}'.format(N_particles, M_val))

# plt.figure()
# r = np.linspace(0.03, 4, 300)
# plt.grid(); plt.yticks([])
# plt.plot(r, basis[0](r), label='$H_1$')
# plt.plot(r, basis[1](r), label='$H_2$')
# plt.legend()

# plot_radials(vecs[0], slaters_m, basis, n, labeler='Ground')
# plt.xlabel('Radius (r)'); plt.ylabel('Valence Density')
# plt.grid()
# plt.legend()
