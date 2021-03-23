import numpy as np
from add_main import *
import pickle


##############################################################################
##### 1-D Harmonic Potential with N particles

# Generates all 2-body and 1-body elements for given number of states
##############################################################################

#Number of single-particle states
N_sp = 11

# Quantum numbers and Basis
n = np.arange(N_sp)
basis = [harmonic_basis(ni) for ni in n]

# Create 2-body and 1-body matrix elements
elements2 = create_elements2(N_sp, basis) # <ij|V|kl>
elements1 = create_elements1(N_sp, basis) # <i|h|j>
elements0 = np.array([elements1.pop((k,k)) for k in range(N_sp)]) # <i|h|i>

# Store elements
filer = "dump{}.pickle".format(N_sp)
with open(filer, "wb") as f:
    pickle.dump(elements2, f)
    pickle.dump(elements1, f)
    pickle.dump(elements0, f)