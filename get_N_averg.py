import math

import numpy as np
from RadiationField import QuantumOscillators_parallel, mpiutil
import h5py
import time

omega_list = list(np.array([2e6, 1e6],dtype=int))

def get_C(chimax, c):
    avrg_E = chimax/2
    cmax = math.sqrt(avrg_E)
    if c <= cmax:
        result = math.sqrt(0.5*(avrg_E - c**2))
    else:
        raise ValueError
    return result

def get_c_list(chimax, c):
    return [get_C(chimax,c),c]

chimax=3999
c_list_1 = get_c_list(chimax, math.sqrt(3999/2))
c_list_2 = get_c_list(chimax, 30)
c_list_3 = get_c_list(chimax, 15)
c_list_4 = get_c_list(chimax, 0)

path = "/data/zzhang/"
#path = "/Users/zheng/Dropbox/project with Nick/"
t1 = time.time()
sys1 = QuantumOscillators_parallel.two_osci_solved(3999, c_list_1, chimax, 1e-16, path)
t2 = time.time()
N_averg_1 = sys1.N_averg_evolution(0,1000,1)
t3 = time.time()

eig_vecs=sys1.eig_vecs
eig_vals=sys1.eig_vals
del sys1
sys = QuantumOscillators_parallel.two_osci_continue(omega_list, c_list_2, chimax, 1e-16, path, eig_vecs, eig_vals)
N_averg_2 = sys.N_averg_evolution(0,1000,1)
sys = QuantumOscillators_parallel.two_osci_continue(omega_list, c_list_3, chimax, 1e-16, path, eig_vecs, eig_vals)
N_averg_3 = sys.N_averg_evolution(0,1000,1)
sys = QuantumOscillators_parallel.two_osci_continue(omega_list, c_list_4, chimax, 1e-16, path, eig_vecs, eig_vals)
N_averg_4 = sys.N_averg_evolution(0,1000,1)


if mpiutil.rank0:
    print("time:{}".format([t2-t1,t3-t2]))
    with h5py.File(path+"N_averg.hdf5", "w") as f:
        f.create_dataset("N_sys1",data=N_averg_1)
        f.create_dataset("N_sys2", data=N_averg_2)
        f.create_dataset("N_sys3", data=N_averg_3)
        f.create_dataset("N_sys4", data=N_averg_4)
