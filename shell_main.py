import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib.colors import BASE_COLORS as colour
from matplotlib.lines import Line2D as mark
from numpy import linalg
from numpy import binary_repr
from tqdm import tqdm
from add_main import *
import pickle

##############################################################################
##### 1-D Harmonic Potential with N particles

# Returns Eigen-everything from CI for N particles and total angular momentum M

# N_paticles must be less than or equal to number of single-particle states:
# N_particles < N_sp
##############################################################################

# Main function
def main(N_particles, N_sp):
    # Number of particles
    
    with open("dump/dump11-2-2w.pickle", "rb") as f:
        elements2 = pickle.load(f)
    
    with open("dump/dump130-harmonic2.pickle", "rb") as f:
        elements1 = pickle.load(f)
        elements0 = pickle.load(f)
    
    for key in list(elements2.keys()):
        for i in key: 
            if i >= N_sp:
                del elements2[key]
                break
    
    elements0 = elements0[:N_sp]
    for key in list(elements1.keys()):
        for i in key: 
            if i >= N_sp:
                del elements1[key]
                break
        
    # Quantum numbers and Basis
    basis = [harmonic_basis(n) for n in range(N_sp)]
    
    # Create possible Slater determinants
    slaters_m = create_slaters(N_sp, N_particles)
    slater_energies = slater_energ(elements0, slaters_m)
    
    sorter = np.argsort(slater_energies)
    slater_energies = slater_energies[sorter]
    slaters_m = np.array(slaters_m)[sorter]
    
    # Build Hamiltonian matrix
    # Add single-particle energies to diagonal elements
    H = np.diag(slater_energies)
    opers = [1 << (N_sp-1-x) for x in range(N_sp)]
    slaters_int = [int(slate, 2) for slate in slaters_m]
    
    scaler = 2.0 # 2-body element interaction scaling factor
    
    for i, slate_main in tqdm(enumerate(slaters_int)):
        slatebin = slaters_m[i]
        
        # 1-body
        for (creat, destr), inter in elements1.items():
            if slatebin[destr] == '1' and slatebin[creat] == '0':
                
                slate = slate_main ^ (opers[destr] + opers[creat])

                phase = slatebin[creat:destr].count('1')
                phase = 1 if phase & 1 == 0 else -1
                
                j = slaters_int.index(slate)
                H[i,j] += inter * phase
            
        # 2-body
        # for (creat1, creat2, destr1, destr2), inter in elements2.items():
        #     if slatebin[destr1] == slatebin[destr2] == '1':
        #         slate = slate_main ^ (opers[destr1] + opers[destr2])
                
        #         phaser = slatebin[destr1:destr2].count('1')
        #         slatebin = binary_repr(slate, N_sp)
                
        #         if slatebin[creat1] == slatebin[creat2] == '0':
        #             slate ^= opers[creat1] + opers[creat2]
        
        #             phaser += slatebin[creat1:creat2].count('1')
        #             phase = 1 if phaser & 1 == 0 else -1
        
        #             j = slaters_int.index(slate)
        #             H[i,j] += scaler * inter * phase
                
    # Symmetrize
    H = H + H.T - np.diag(H.diagonal())
    #plt.imshow(H)
    
    # Diagonalize the Hamiltonian matrix
    eigs, vecs = linalg.eig(H)
    vecs = vecs.T
    order = np.argsort(eigs)
    eigs, vecs = eigs[order], vecs[order]
    return eigs, vecs, slaters_m
        
#################################
#
#%%

N_particles = 1
eigen = []
N = range(4, 11)
excit = 4

for Nsp in N:
    eigs, vecs, slaters = main(N_particles, Nsp)
    eigen.append(eigs[:excit])
    vecs = np.square(vecs)

eigen = np.array(eigen)
# plt.figure()
# plt.bar(np.arange(len(slaters)), vecs[0])
# plt.xlabel('Slater no.')
# plt.ylabel('Coefficient$^2$')
# print(eigs[:5])

#%%

### Hamronic (2, 1) - one-body only
true = sqrt(2) * (np.arange(excit) + 0.5)
#true = np.array([0.11663698, 0.46060594, 1.01698817, 1.76716169, 2.69177977])

eigen = (abs(eigen-true) / true * 100).T
colour = list(colour.keys())
mark = list(mark.markers.keys())

for n, row in enumerate(eigen):
    strin = str(n) + '-excited' if n != 0 else 'ground'
    plt.plot(N, np.log(row), mark[n+2], label=strin, markersize=4, c=colour[n])
    plt.plot(N, np.log(row), c=colour[n])

plt.title('Woods-Saxon potential')
plt.legend()
plt.xlabel('Number of basis states used')
plt.ylabel('log(error %)')



