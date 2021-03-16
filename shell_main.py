import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg
from timeit import default_timer as timer
from tqdm import tqdm
from add_main import *


##############################################################################
##### 1-D Woods Saxon Potential with N particles

# Returns Eigen-everything from CI for N particles and total angular momentum M

# N_paticles must be less than or equal to number of single-particle states:
# N_particles > N_sp

# total 2*M of the system must be same parity as number of particles:
# (N_particles & 1) != (M_val & 1)
##############################################################################



# Number of particles and Number of single-particle states
N_particles = 5
N_sp = 12

# Quantum numbers and Basis
n = np.arange(1, N_sp+1)
basis = [Harmonic.basis(ni) for ni in n]
sp_energies = np.array([Harmonic.energies(ni) for ni in n])

# Create possible Slater determinants
slaters_m = create_slaters(N_sp, N_particles)
slater_energies = slater_energy(slaters_m, sp_energies)
N_slater = len(slaters_m)

sorter = np.argsort(slater_energies)
slater_energies = slater_energies[sorter]
slaters_m = slaters_m[sorter]

# Create matrix combinations and elements
matrix_combs = create_matrix_combs(n)
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
        
#################################
# Main Function
