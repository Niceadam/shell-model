#============================
# Additional functions 
#=============================

import numpy as np
import matplotlib.pyplot as plt
import itertools
import copy

#==================================================================
# This function permutates array and gives the phase +1 or -1
# depending on the amount of moves in permutation
# RETURNS sign from the permutation and the ordered array
def permutator(array):
    alist = array.tolist()
    nummov=0 #number of moves
    #------------------------------------------------------
    # at first we check that there are no two same elements
    seen = []
    for number in alist:
        if number in seen:
            #print "===================================================="
            #print "             E R R O R !!!!                     "
            #print " Pauli is angry - two particles on the same state!"
            #print "===================================================="
            return 0,[]  
        else:
            seen.append(number)
       #The list 'seen' not needed later
    #--------------------------------------------------------
    # PERMUTATOR ITSELF
    #--------------------------------------------------------
    if len(alist)!=0:
       for i in range(0, len(alist)):
           minval=min(alist[i:]) # find min. value between i:th position
                                 # and the last (first i states have already
                                 # been permutated
           minindex=alist.index(minval) # store the index of minval
           nummov=nummov+(minindex-i) #update number of moves needed
           alist.remove(minval) #remove minval from the list
           alist.insert(i,minval) #insert minval in the position i
       ordarray=np.asarray(alist)
       # Even number of moves -> +1, odd number of moves -> -1
       if nummov%2==0:
           sign=1
       else: 
           sign=-1
       return sign,ordarray
    # if len(alist) == 0 then:
    else:
       return 0,array #array is empty

#======================================================================
# LEVEL PLOTTER: plots the given eigenenergies
#======================================================================
def showlevels(energies):
    for e in energies:
        plt.plot((1.0,1.5),(e,e),'k-') #draws a line at energy e
    plt.plot((0.5,2.0),(0,0),'r--')
    plt.ylabel('Energy')
    plt.title('Energy levels')
    plt.xlim([0,2.5])
    plt.tick_params(       # Remove x-tics:
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off') # labels along the bottom edge are off
    plt.show()
    return 


#==============================================================================================
# Routine that picks only those slater
# determinants of the pairing problem
#
# E.g. with input:
#
# myslaters=[[1,2,3,4],[1,2,4,5],[3,4,5,8],[1,2,7,8],[3,5,6,7], [3,4,5,6], [3,4,7,8], [1,6,7,8]]
#
# you will get
#
# [[1, 2, 3, 4], [1, 2, 7, 8], [3, 4, 5, 6], [3, 4, 7, 8]]
#==============================================================================================
def pickpairs(allslaters):
   notsuitable = np.zeros(len(allslaters)) #this is just an array for marking 0=fine, 1=not suitable sd
   for sd in allslaters:
      sdind=allslaters.index(sd) #index for the slater determinant which is studied now
      if len(sd)%2 != 0:         # check that there are even number of particles now
         print("Number of particles is not even!")
         return []
      halflen=len(sd)/2         # I want to loop over 2*i so I have to take a half of the upper limit
      for i in range(0,halflen):
          if sd[2*i]%2 == 0:    # First, third... particle must be on an odd state 
             notsuitable[sdind]=1
          elif sd[2*i+1]%2 !=0: # Second, fourth... particle must be on an even state
             notsuitable[sdind]=1
          elif sd[2*i] != sd[2*i+1]-1: # Because we have pairs and sd's are ordered, 1st and 2nd particle must be 1,2 or 3,4 etc
             notsuitable[sdind]=1
   suitableslaters = []
   for ind in range(0,len(notsuitable)):
       if (notsuitable[ind] == 0):
          suitableslaters.append(allslaters[ind]) # finally we combine suitable slater determinants
   return suitableslaters


#==============================================================================================
# Routine that picks only those slater
# determinants of the pairing problem
#
# E.g. with input:
#
# n=np.array([0,0,1,1,2,2,3,3])
# l=np.array([0,0,0,0,0,0,0,0])
# j=np.array([1,1,1,1,1,1,1,1])
# myslaters=[[1,2,3,4],[1,2,4,5],[3,4,5,8],[1,2,7,8],[3,5,6,7], [3,4,5,6], [3,4,7,8], [1,6,7,8]]
#
# you will get
#
# [[1, 2, 3, 4], [1, 2, 7, 8], [3, 4, 5, 6], [3, 4, 7, 8]]
#
# TODO Think about m values: do we need constraint ???
#==============================================================================================
def pickpairsnljm(allslaters,n,l,j):
   notsuitable = np.zeros(len(allslaters)) #this is just an array for marking 0=fine, 1=not suitable sd
   for sd in allslaters:
      sdind=allslaters.index(sd) #index for the slater determinant which is studied now
      if len(sd)%2 != 0:         # check that there are even number of particles now
         print("Number of particles is not even!")
         return []
      halflen=len(sd)/2         # I want to loop over 2*i so I have to take a half of the upper limit
      for i in range(0,halflen):
         if n[sd[2*i]-1] != n[sd[2*i+1]-1] or l[sd[2*i]-1] != l[sd[2*i+1]-1] or j[sd[2*i]-1] != j[sd[2*i+1]-1]:
            notsuitable[sdind]=1
   suitableslaters = []
   for ind in range(0,len(notsuitable)):
       if (notsuitable[ind] == 0):
          suitableslaters.append(allslaters[ind]) # finally we combine suitable slater determinants
   return suitableslaters


#==============================================================================================
# Picks the slaterdeterminants which have given M
# This code calculates the sum of m values of particles in a slaterdeterminant, and if
# the sum is equals to given M, slater determinant is accepted
#
# For example, with given input
# myslaters=[[1,2,3,4],[1,3,5,7],[3,5,6,7],[1,3,7,8],[2,4,6,8], [3,4,5,6], [4,5,6,7], [1,6,7,8]]
# m=np.array([-1,1,-1,1,-1,1,-1,1])
# M=0
#
# you get
# [[1, 2, 3, 4], [3, 4, 5, 6], [4, 5, 6, 7], [1, 6, 7, 8]]
#==============================================================================================
def pickslatersM(allslaters,m,totalM):
   totalMs=[]
   suitableslaters=[]
   for sd in allslaters:
      sdind=allslaters.index(sd) # index for the sd which is studied
      sdM=0
      for i in range(0,len(sd)): # sum all ms
          sdM=sdM+m[sd[i]-1]
      if sdM == totalM:
          suitableslaters.append(allslaters[sdind])
   return suitableslaters


#==============================================================================================
# Functions that returns a set of set of possible 
# Slater determinant for set of states s and m number of particles
#==============================================================================================
def find_slaters(s,m):
	return set(itertools.combinations(s,m))


#==============================================================================================
# Function that takes two-body operator indices p,q,r,s and index alpha for the ket Slater determinant 
# and returns the index beta of the new Slater determinant and the phase
#==============================================================================================
def two_body(p,q,r,s,alpha,sd_pairs2):
	if (r in sd_pairs2[alpha]) and (s in sd_pairs2[alpha]) and (p != q) and (r !=s) and (p == r) and (q == s):
		slater= np.array(sd_pairs2[alpha])
		(sign,slater3) = permutator(slater)
		slater3 = tuple(slater3)
		if slater3 in sd_pairs2:
			beta = sd_pairs2.index(slater3)
		else:
			beta = 0
			sign = 0
		return (beta,sign)
	elif (r in sd_pairs2[alpha]) and (s in sd_pairs2[alpha]) and (p != q) and (r !=s):
		# Replace the r- and s-values in alpha with p and q.
		slater = list(copy.deepcopy(sd_pairs2[alpha]))
		sd_pairs2 = list(sd_pairs2)
		slater2 = list(copy.deepcopy(slater))
		slater2[slater.index(s)] = q
		slater2[slater.index(r)] = p
		slater2 = np.array(slater2)
		(sign,slater3) = permutator(slater2)
		slater3 = tuple(slater3)
		if slater3 in sd_pairs2:
			beta = sd_pairs2.index(slater3)
		else:
			beta = 0
			sign = 0
		return (beta,sign)
	else:
		# new Slater determinant is zero (set phase to zero and index to N_slater+10)
		beta = int(len(sd_pairs2)+10)
		return (beta,0)
    
    
#==============================================================================================
# Function that calculates a diagonal element in the Hamiltonian by summing the 
# single-particle energies of the Slater determinant
#==============================================================================================
def diag_element(slater,energies,N_part):
	diag_energy = 0
	for i in range(N_part):
		particle = slater[i]
		diag_energy += energies[particle-1]
	return diag_energy
