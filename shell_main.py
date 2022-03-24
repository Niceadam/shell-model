import numpy as np
from math import sqrt
import matplotlib as matt
import matplotlib.pyplot as plt
from matplotlib.colors import BASE_COLORS as colour
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
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
    
    with open("dump/dump13-2-normal.pickle", "rb") as f:
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
    
    scaler = 1 / sqrt(pi) # 2-body element interaction scaling factor
    scaler *= 1
    
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
        for (creat1, creat2, destr1, destr2), inter in elements2.items():
            if slatebin[destr1] == slatebin[destr2] == '1':
                slate = slate_main ^ (opers[destr1] + opers[destr2])
                
                phaser = slatebin[destr1:destr2].count('1')
                slatebin = binary_repr(slate, N_sp)
                
                if slatebin[creat1] == slatebin[creat2] == '0':
                    slate ^= opers[creat1] + opers[creat2]
        
                    phaser += slatebin[creat1:creat2].count('1')
                    phase = 1 if phaser & 1 == 0 else -1
        
                    j = slaters_int.index(slate)
                    H[i,j] += scaler * inter * phase
                
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

#%%

# Slater Profiles

N_particles = 11
N_sp = 13

eigs, vecs0, slaters = main(N_particles, N_sp)
vecs = np.square(vecs0)

cmap = ['blue', 'green', 'orange', 'red']
fig = plt.figure()
ax = fig.gca(projection='3d')

lim = 18
for i in [0, 1, 2, 3]:
    x = np.arange(len(slaters))[:lim]
    y = i
    dz = vecs[i][:lim]
    ax.bar3d(x, y, 0, 0.7, 0.18, dz, color=cmap[i], shade=True)

ax.set_xlabel('Determinant number')
ax.set_ylabel('Excitation')
ax.set_zlabel('Magnitude of Coefficient')
ax.set_yticks([0, 1, 2, 3])


#%%

eigen = []
excit = 5
N = range(excit, N_sp)
for Nsp in N:
    eigs, vecs, slaters = main(N_particles, Nsp)
    eigen.append(eigs[:excit])

eigenm = np.array(eigen)

#%%

N_particles = 10
N_sp = 13

eigs, vecs0, slaters = main(N_particles, N_sp)
vecs = np.square(vecs0)

# For 1 particle
plt.figure()
x = np.linspace(-6, 6, N_sp)
basis = [harmonic_basis(n, 1)(x) for n in range(N_sp)]

state = 2
wave = sum(base * vec for base, vec in zip(basis, vecs[state])) 

plt.plot(x, wave, label='Computed')
plt.xlabel("x", size=18)
plt.ylabel("$\Psi$", size=18)
    
plt.plot(x, harmonic_basis(state, 2)(x), label="Analytical")
plt.yticks([])
plt.legend()

#%%
# Density plots

# # For 2 particle
# fig = plt.figure()
# ax = fig.gca(projection='3d')

# x = np.linspace(-4, 4, 200)
# x1, x2 = np.meshgrid(x, x)

# basis_x1 = [harmonic_basis(n, 1)(x1) for n in range(20)]
# basis_x2 = [harmonic_basis(n, 1)(x2) for n in range(20)]
# wave = np.zeros((200, 200))

# def slaterfunc(i, j):
#     return basis_x1[i] * basis_x2[j] - basis_x1[j] * basis_x2[i]

# state = 0
# for slater, coeff in tqdm(zip(slaters, vecs0[state])):
#     i, j = slater.find('1'), slater.rfind('1')
#     wave += coeff * slaterfunc(i, j)

# #density /= np.sqrt(2)
# density = wave**2

# ax.plot_surface(x1, x2, density, cmap=cm.gnuplot)
# ax.view_init(30, 50)
# ax.set_xlabel('x1')
# ax.set_ylabel('x2')
# ax.set_zticks([])
# plt.figure()


#%%

# ### Harmonic (2, 1) - one-body only
#true = sqrt(2) * (np.arange(excit) + 0.5)
# #### Harmonic (1, 1)- one-body only
#true = (np.arange(excit) + 0.5)
#print(sum(true))
# Woods-Saxon
#true = np.array([0.11663698, 0.46060593, 1.01698817, 1.76716165, 2.69177977])

# Harmonic (1, 1) - 2-body normal
#true = np.array([1.65240159, 2.65243612, 4.        , 4.0267893 , 5.        ])

# Harmonic2 - 13 states
#true = np.array([2.68238388, 4.10073489, 5.65300188, 5.70059422, 7.06053617])
# Harmonic2 - 15 states - 0.1w
#true = np.array([2.68910518, 4.10723662, 5.65489449, 5.66216099, 7.0653616 ])
# 15 states - 2w
#true = np.array([2.7249395 , 4.14214998, 5.65363939, 5.69466124, 7.06234717])

# Woods - 13 states - 1
#true = np.array([0.55043791, 1.08656227, 1.45083559, 1.86521063, 2.23460314])
# Woods - 15 states - 0.1w
#true = np.array([0.50179537, 1.04039784, 1.45649078, 1.86111692, 2.23083541])
# Woods - 15 states - 2w
# true = 

# Woods-corrug - 100 states - 1
#true = np.array([0.11663698, 0.46060563, 1.01698817, 1.76716161, 2.69177977])

# 13 states - 2m - 2body
#true = np.array([2.59689061, 4.01772065, 5.58111776, 5.6623937 , 6.98309154])
# 40 states- 2m -  1body
true = eigenm[-1][:]


eigen = (abs(eigenm-true) / true * 100).T
colour = list(colour.keys())
mark = list(mark.markers.keys())

for n, row in enumerate(eigen[:-1]):
    strin = str(n) + '-excited' if n != 0 else 'ground'
    plt.plot(N, np.log(row), mark[n+2], label=strin, markersize=2, c=colour[n])
    plt.plot(N, np.log(row), c=colour[n])

plt.title('')
plt.legend()
plt.xlabel('Number of basis states used')
plt.ylabel('log(error %)')



