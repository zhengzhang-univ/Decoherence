import numpy as np
from RadiationField import QuantumOscillators_parallel, mpiutil
import h5py
import time
st = time.time()

omega_list = list(np.array([2e6, 1e6],dtype=int))
c_list_1 = [35,35]
c_list_2 = [15,np.sqrt(3675 - 2*15**2)]
c_list_3 = [5,np.sqrt(3675 - 2*5**2)]
c_list_4 = [0.5*np.sqrt(3675 - 5**2), 5]


def heavist_chi(c_list):
    return int(2*np.absolute(c_list[0])**2 + np.absolute(c_list[1])**2)
enrg =  heavist_chi(c_list_1)
path = "/data/zzhang/"
#path = "/Users/zheng/Dropbox/project with Nick/"

sys1 = QuantumOscillators_parallel.two_osci_solved(omega_list, c_list_1, enrg*2, 1e-16, path)
N_averg_1 = sys1.N_averg_evolution(0,1000,1)
eig_vecs=sys1.eig_vecs
eig_vals=sys1.eig_vals
del sys1
sys = QuantumOscillators_parallel.two_osci_continue(omega_list, c_list_2, enrg*2, 1e-16, path, eig_vecs, eig_vals)
N_averg_2 = sys.N_averg_evolution(0,1000,1)
sys = QuantumOscillators_parallel.two_osci_continue(omega_list, c_list_3, enrg*2, 1e-16, path, eig_vecs, eig_vals)
N_averg_3 = sys.N_averg_evolution(0,1000,1)
sys = QuantumOscillators_parallel.two_osci_continue(omega_list, c_list_4, enrg*2, 1e-16, path, eig_vecs, eig_vals)
N_averg_4 = sys.N_averg_evolution(0,1000,1)



et = time.time()
if mpiutil.rank0:
    print("Elapsed time (s): {}".format(et - st))
    with h5py.File(path+"N_averg.hdf5", "w") as f:
        f.create_dataset("N_sys1",data=N_averg_1)
        f.create_dataset("N_sys2", data=N_averg_2)
        f.create_dataset("N_sys3", data=N_averg_3)
        f.create_dataset("N_sys4", data=N_averg_4)
