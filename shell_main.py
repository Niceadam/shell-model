import numpy as np
#import matplotlib.pyplot as plt
from numpy import linalg
from numpy import binary_repr
from tqdm import tqdm
from add_main import *
import pickle


##############################################################################
##### 1-D Harmonic Potential with N particles

# Returns Eigen-everything from CI for N particles and total angular momentum M

# N_paticles must be less than or equal to number of single-particle states:
# N_particles > N_sp

# total 2*M of the system must be same parity as number of particles:
# (N_particles & 1) != (M_val & 1)
##############################################################################

#Number of single-particle states
N_sp = 6

# Quantum numbers and Basis
n = np.arange(N_sp)
basis = [harmonic_basis(ni) for ni in n]

# Create 2-body and 1-body matrix elements
elements2 = create_elements2(N_sp, basis) # <ij|V|kl>
elements1 = create_elements1(N_sp, basis) # <i|h|j>
elements0 = np.array([elements1.pop((k,k)) for k in range(N_sp)]) # <i|h|i>

# # # Store elements
# filer = "dump{}.pickle".format(N_sp)
# with open(filer, "wb") as f:
#     pickle.dump(elements2, f)
#     pickle.dump(elements1, f)
#     pickle.dump(elements0, f)

#%%
# Read elements
# filer = "dump{}.pickle".format(N_sp)
# with open(filer, "rb") as f:
#     elements2 = pickle.load(f)
#     elements1 = pickle.load(f)
#     elements0 = pickle.load(f)

#%%

# Number of particles
N_particles = 4

# Create possible Slater determinants
slaters_m = create_slaters(N_sp, N_particles)
slater_energies = slater_energ(elements0, slaters_m)
N_slater = len(slaters_m)

sorter = np.argsort(slater_energies)
slater_energies = slater_energies[sorter]
slaters_m = np.array(slaters_m)[sorter]

# Build Hamiltonian matrix
# Add single-particle energies to diagonal elements
H = np.diag(slater_energies)
opers = [2**(N_sp-1-x) for x in range(N_sp)]
slaters_int = [int(slate, 2) for slate in slaters_m]

for i, slate_main in tqdm(enumerate(slaters_int)):
    slatebin = slaters_m[i]
    
    # 1-body
    for (creat, destr), inter in elements1.items():
        if slatebin[destr] == '1' and slatebin[creat] == '0':
            slate = slate_main ^ (opers[destr] + opers[creat])
            phaser = slatebin[:destr].count('1')
            phaser += slatebin[:creat].count('1')
            if destr < creat: phaser -= 1
                
            phase = 1 if phaser % 2 == 0 else -1
            
            j = slaters_int.index(slate)
            H[i,j] += inter * phase
        
    # 2-body
    for (creat1, creat2, destr1, destr2), inter in elements2.items():
        if slatebin[destr1] == slatebin[destr2] == '1':
            slate = slate_main ^ (opers[destr1] + opers[destr2])
            phaser = slatebin[:destr2].count('1')
            phaser += slatebin[:destr1].count('1')
            slatebin = binary_repr(slate, N_sp)
            
            if slatebin[creat1] == slatebin[creat2] == '0':
                slate ^= opers[creat1] + opers[creat2]
                phaser += slatebin[:creat2].count('1')
                phaser += slatebin[:creat1].count('1')
                phase = 1 if phaser % 2 == 0 else -1
                
                j = slaters_int.index(slate)
                H[i,j] += inter * phase
            
# Symmetrize
H = H + H.T - np.diag(H.diagonal())

# Diagonalize the Hamiltonian matrix
eigs, vecs = linalg.eig(H)
vecs = vecs.T
order = np.argsort(eigs)
eigs, vecs = eigs[order], vecs[order]
        
#################################
# Main Function

# vecs = np.square(vecs)
# plt.bar(np.arange(N_slater), vecs[0])
