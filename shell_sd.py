# Nuclear TALENT 2017
# Tiia, Shane, Martin, Ovidiu
# This code reads the single-particle states and effective interactions 
# for N particles and total angular momentum M and plots the energy eigenvalues

# EDIT: NEW SHELL CODE - Adam Asaad

import numpy as np
import matplotlib as plt2
import matplotlib.pyplot as plt
from numpy import linalg
from timeit import default_timer as timer

from add_funcs_edit import *
from read_lpt import read_lpt, read_lpt_exp

# Load single-particle data from file 'sdshellint.dat' ordered as (index, n, l, 2j, 2m_j)
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

index1 = int_data[:,0]
index2 = int_data[:,1]
index3 = int_data[:,2]
index4 = int_data[:,3]

def main_eigen(N_particles, M_val):    
    """
    Returns Eigen-everything from CI for N particles and total angular momentum M"""
    
    # N_paticles must be less than or equal to number of single-particle states)
    if N_particles > N_sp:
        raise "Wrong input: Number of particles must be less than number of single-particle states"
    
    # total 2*M of the system must be same parity as number of particles
    if (N_particles & 1) != (M_val & 1):
        raise 'Matching Parity'
    
    
    # Number of matrix elements of effective interaction + scaling factor
    matrix_elements = int_data[:,4]
    scaling_factor = (18.0 / (16.0+N_particles))**0.3
    matrix_elements = scaling_factor * matrix_elements
    
    # Create possible Slater determinants with given 2*M
    slaters_m = create_slaters(N_sp, N_particles, m_j, M_val)
    N_slater = len(slaters_m)
    
    # Build Hamiltonian matrix
    H = np.zeros((N_slater, N_slater))
    for beta in range(N_slater):
        # Add single-particle energies to diagonal elements
        H[beta, beta] += diag_element(slaters_m[beta], sp_energies)
        for a in range(len(matrix_elements)):
            (alpha, phase) = two_body(index1[a],index2[a],index3[a],index4[a],beta,slaters_m)
            if phase != 0:
                elem = int(phase) * matrix_elements[a]
                if alpha == beta:
                    H[beta, beta] += elem
                else:
                    H[alpha, beta] += elem
                    H[beta, alpha] += elem
    
    # Diagonalize the Hamiltonian matrix
    eigs, vecs = linalg.eig(H)
    vecs = vecs.T
    
    return eigs, vecs, slaters_m
    
def plot_energies(eigs):
    """Gives the energy plot for given Eigen-energies"""
    
    
    # Include only levels below mev_limit MeV
    mev_lim = 6
    min_energy = eigs.min()
    E = eigs - min_energy
    E = E[E<mev_lim]
    
    plt2.rc('xtick', labelsize=20)
    plt2.rc('ytick', labelsize=20)
    
    if (N_particles % 2) == 0:
       # Compare with NushellX levels
       E_nu = read_lpt(N_particles)
       E_nu = E_nu[E_nu<mev_lim]
    
       E_exp = read_lpt_exp(N_particles)
       E_exp = E_exp[E_exp<mev_lim]
    
       for e in E_nu:
           plt.plot([1.5,2.5],[e,e],'r')
       for e in E_exp:
           plt.plot([-1.5,-0.5],[e,e],'k')
    
    for e in E:
    	plt.plot([0,1],[e,e],'b')
    
    plt.ylabel('Energy (MeV)', fontsize=20)
    plt.xticks([-1,0.5,2.0],['Exp','Calc','NushellX'])
    plt.ylim(-0.25, mev_lim)
    plt.title(r'$^{'+str(16+N_particles)+'}$'+'O', fontsize=20)
    plt.grid()
    
def plot_radials(weights, slaters_m, rlim=[0.01, 3], dim='1D', labeler='Ground'):
    """Given possible Slaters + CI weights for each slater = Plot Radial Desntiy function
    
    Only 2 3D Radial Functions: 
        1d: n=0 l=2
        2s: n=1 l=0
    """
    
    global N_particles, M_val
    
    # Setup grid
    resol = 300 # resolution for the arrays used for plotting
    r = np.linspace(rlim[0], rlim[1], resol)
    
    # Evaluate orbital basis radials for each possible state
    if dim.upper() == '3D':
        basis = [create_radial_3D(n[k], l[k])(r) for k in range(N_sp)]
    elif dim.upper() == '1D':
        basis = [create_radial_1D(n[k])(r) for k in range(N_sp)]
    
    # Final density = sum(weight[i]**2 * slater_dens[i])
    radial_dens = np.zeros(resol)
    for i, slate in enumerate(slaters_m):
        
        # Calculate Slater Density. You can't plot Slaters directly obviously!!
        slate_den = sum(basis[k-1]**2 for k in slate)
                        
        # Add Slate density to Global Density with ground weight |c_i|^2
        radial_dens += weights[i]**2 * slate_den

    plt.plot(r, radial_dens, label=labeler)
    
#################################
# Main Function

N_particles, M_val = 6, 8
eigs, vecs, slaters_m = main_eigen(N_particles, M_val)

plot_radials(vecs[0], slaters_m, dim='1D', labeler='Ground')
plt.xlabel('Radius (r)'); plt.ylabel('Density')
plt.grid()
plt.legend()