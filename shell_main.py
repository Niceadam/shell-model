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
N_particles = 4
N_sp = 6

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

for i, slate1 in tqdm(enumerate(slaters_m)):
    for j, slate2 in enumerate(slaters_m[i:]):
        comber = np.binary_repr(int(slate1, 2) ^ int(slate2, 2), N_sp)
        if comber.count('1') == 4:
            H[i,j] += matrix_element(comber, basis)
 
# Symmetrize
H += H.T - np.diag(H.diagonal())
plt.imshow(H)
          
# Diagonalize the Hamiltonian matrixs
eigs, vecs = linalg.eig(H)
vecs = vecs.T
order = np.argsort(eigs)
eigs, vecs = eigs[order], vecs[order]
        
#################################
# Main Function
