import numpy as np
from add_main import *
from time import time
import pickle

##############################################################################
##### 1-D Harmonic Potential with N particles

# Generates all 2-body and 1-body elements for given number of states
##############################################################################

#%%
#Number of single-particle states
N_sp = 14

# Quantum numbers and Basis
basis = [harmonic_basis(n) for n in range(N_sp)]

# 2-body - for given interaction
with open("dump/dump13-2-2w-2m.pickle".format(N_sp), "rb") as f:
    elements2 = pickle.load(f)

start = time()
adder = create_elements2(N_sp, basis)
elements2.update(adder)
print(time() - start)

with open("dump/dump{}-2-2w-2m.pickle".format(N_sp), "wb") as f:
    pickle.dump(elements2, f)
    
#%%
## 1-body - for given potential

N_sp = 100
basis = [harmonic_basis(n) for n in range(N_sp)]

# with open("dump/dump150-woods.pickle".format(N_sp), "rb") as f:
#     elements1 = pickle.load(f)
#     elements0 = pickle.load(f)

elements1 = create_elements1(N_sp, basis) # <i|h|j>
elements0 = [elements1.pop((k,k)) for k in range(N_sp)]

# elements1.update(dicter)
# elements0[(N_sp-1, N_sp-1)] = dicter.pop((N_sp-1, N_sp-1))
#%%
with open("dump/dump{}-woodscorrug.pickle".format(N_sp), "wb") as f:
    pickle.dump(elements1, f)
    pickle.dump(elements0, f)