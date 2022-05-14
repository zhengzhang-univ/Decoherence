import math

import numpy as np
from RadiationField import QuantumOscillators_parallel, mpiutil
import h5py
import time

omega_list = list(np.array([2e6, 1e6],dtype=int))

def get_C(chimax, c):
    avrg_E = chimax/2.
    cmax = np.sqrt(avrg_E)
    if c < cmax:
        result = np.sqrt(0.5*(avrg_E - c**2))
    else:
        raise ValueError
    return result

def get_c_list(chimax, c):
    return [get_C(chimax,c),c]

chimax=3999
c_list_1 = get_c_list(chimax, 35)
c_list_2 = get_c_list(chimax, 25)
c_list_3 = get_c_list(chimax, 15)
c_list_4 = get_c_list(chimax, 5)

path = "/data/zzhang/"
#path = "/Users/zheng/Dropbox/project with Nick/"
st, et, dt = 0, 2000, 1

t1 = time.time()
sys = QuantumOscillators_parallel.two_osci_solved(omega_list, c_list_1, chimax, 1e-16, path)
N_averg_1 = sys.N_averg_evolution(st, et, dt)
t2 = time.time()

sys = QuantumOscillators_parallel.two_osci_solved(omega_list, c_list_2, chimax, 1e-16, path)
N_averg_2 = sys.N_averg_evolution(st, et, dt)

sys = QuantumOscillators_parallel.two_osci_solved(omega_list, c_list_3, chimax, 1e-16, path)
N_averg_3 = sys.N_averg_evolution(st, et, dt)

sys = QuantumOscillators_parallel.two_osci_solved(omega_list, c_list_4, chimax, 1e-16, path)
N_averg_4 = sys.N_averg_evolution(st, et, dt)
t3 = time.time()

if mpiutil.rank0:
    print("time:{}".format([t2-t1,t3-t1]))
    ts = np.arange(st, et, dt)
    with h5py.File(path+"N_averg.hdf5", "w") as f:
        f.create_dataset("time",data=ts)
        f.create_dataset("N_sys1",data=N_averg_1)
        f.create_dataset("N_sys2", data=N_averg_2)
        f.create_dataset("N_sys3", data=N_averg_3)
        f.create_dataset("N_sys4", data=N_averg_4)