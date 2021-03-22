import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg
from numpy import binary_repr
from timeit import default_timer as timer
from tqdm import tqdm
from add_main import *


##############################################################################
##### 1-D Harmonic Potential with N particles

# Returns Eigen-everything from CI for N particles and total angular momentum M

# N_paticles must be less than or equal to number of single-particle states:
# N_particles > N_sp

# total 2*M of the system must be same parity as number of particles:
# (N_particles & 1) != (M_val & 1)
##############################################################################

# Number of particles and Number of single-particle states
N_particles = 6
N_sp = 12

# Quantum numbers and Basis
n = np.arange(N_sp)
basis = [Harmonic.basis(ni) for ni in n]
sp_energies = np.array([Harmonic.energies(ni) for ni in n])

# Create possible Slater determinants
slaters_m = create_slaters(N_sp, N_particles)
slater_energies = slater_energy(slaters_m, sp_energies)
N_slater = len(slaters_m)

# Build Hamiltonian matrix
# Add single-particle energies to diagonal elements
H = np.diag(slater_energies)
slaters_int = [int(slate, 2) for slate in slaters_m]

for i, slate1 in tqdm(enumerate(slaters_int)):
    for j, slate2 in enumerate(slaters_int[i:]):
        comber = binary_repr(slate1 ^ slate2, N_sp)
        counter = comber.count('1')
        if counter <= 4:
            if counter == 4: # 2 orbital difference
                H[i,j] += matrix_element2(comber, basis, N_sp)
            elif counter == 2: # 1 orbital difference
                H[i,j] += matrix_element1(comber, basis, N_sp)
            elif counter == 0: # Diagonal
                H[i,j] += matrix_element0(comber, basis, N_sp)
            
# Symmetrize
H = H + H.T - np.diag(H.diagonal())

# Diagonalize the Hamiltonian matrixs
eigs, vecs = linalg.eig(H)
vecs = vecs.T
order = np.argsort(eigs)
eigs, vecs = eigs[order], vecs[order]
        
#################################
# Main Function
