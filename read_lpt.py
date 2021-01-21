import numpy as np
import matplotlib.pyplot as plt


## Reads NushellX calculation outputs
def read_lpt(N_particles):
	data = np.loadtxt('oxygen_data/o_'+str(16+N_particles)+'2.lpt',skiprows=6,usecols=[0,1,2,3,4,5,6])

	number = []
	nth_J = []
	E = []
	E_ex = []
	J = []
	T_z = []
	parity = []

	for datum in data:
		number.append(datum[0])
		nth_J.append(datum[1])
		E.append(datum[2])
		E_ex.append(datum[3])
		J.append(datum[4])
		T_z.append(datum[5])
		parity.append(datum[6])

	number = np.array(number)
	nth_J = np.array(nth_J)
	E = np.array(E)
	E_ex = np.array(E_ex)
	J = np.array(J)
	T_z = np.array(T_z)
	parity = np.array(parity)

	return E_ex

def read_lpt_exp(N_particles):
	data = np.loadtxt('oxygen_data/o_0'+str(16+N_particles)+'exp.lpt',skiprows=1,usecols=[0])

	return data

#print(read_lpt_exp(2))
#print(read_lpt(2))
