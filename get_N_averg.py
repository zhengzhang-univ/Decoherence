import numpy as np
from RadiationField import QuantumOscillators_parallel, mpiutil
import h5py
import time
st = time.time()

omega_list = list(np.array([2e6, 1e6],dtype=int))
c_list_1 = [10,10]
c_list_2 = [15,np.sqrt(3675 - 2*15**2)]
c_list_3 = [5,np.sqrt(3675 - 2*5**2)]
c_list_4 = [0.5*np.sqrt(3675 - 5**2), 5]


def heavist_chi(c_list):
    return int(2*np.absolute(c_list[0])**2 + np.absolute(c_list[1])**2)
enrg =  heavist_chi(c_list_1)
path = "/data/zzhang/"
#path = "/Users/zheng/Dropbox/project with Nick/"

two_ocsi_sys = QuantumOscillators_parallel.two_osci_solved(omega_list, c_list_1, enrg*2, 1e-16, path)
N_averg = two_ocsi_sys.N_averg_evolution(0,1000,1, 'N')

et = time.time()
if mpiutil.rank0:
    print("Elapsed time (s): {}".format(et - st))
    tlist = np.linspace(0,800,1)
    with h5py.File(path+"N_averg.hdf5", "w") as f:
        f.create_dataset("t",data=tlist)
        f.create_dataset("N_sys1",data=N_averg)
