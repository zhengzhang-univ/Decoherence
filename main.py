import numpy as np
from RadiationField import QuantumOscillators, mpiutil
import h5py
#import matplotlib.pyplot as plt
#from RadiationField.visulization import plots_2d_in_3d, animation_imshow
import time
st = time.time()
QuantumOscillators.solve_whole_system_and_save_3(1000)
et = time.time()
print("Elapsed time (s): {}".format(et-st))

"""
omega_list = list(np.array([2e6, 1e6],dtype=int))

def heavist_chi(c_list):
    return int(2*np.absolute(c_list[0])**2 + np.absolute(c_list[1])**2)

c_list_1 = [30,30]
two_ocsi_sys_1 = QuantumOscillators.Chi_analysis(omega_list, c_list_1, 2700, 1e-16)
#N_and_Nchi_ts_1, C_chi_1, tlist = two_ocsi_sys_1.Average_N_decomposed_evolution(0,10000,1)
dm_array, N_averg = two_ocsi_sys_1.density_matrix_evolution(0,1000,1, 'N')
if mpiutil.rank0:
    tlist = np.linspace(0,1000,1)
    with h5py.File("oscillators_1.hdf5", "w") as f:
        f.create_dataset("t",data=tlist)
        f.create_dataset("dm_array",data=dm_array)
        f.create_dataset("N",data=N_averg)
        f.create_dataset("c",data=c_list_1)
        f.create_dataset("chimax", data=two_ocsi_sys_1.Chimax)
"""

"""
n_2N_avg = 2*c_list_1[0]**2 + c_list_1[1]**2
c_list_2 = [np.sqrt(n_2N_avg-100**2),100]
c_list_3 = [100,np.sqrt(n_2N_avg-2*100**2)]


two_ocsi_sys_2 = QuantumOscillators.Chi_analysis(omega_list, c_list_2, 20000, 1e-16)
N_and_Nchi_ts_2, C_chi_2, tlist = two_ocsi_sys_2.Average_N_decomposed_evolution(0,10000,1)

two_ocsi_sys_3 = QuantumOscillators.Chi_analysis(omega_list, c_list_3, 20000, 1e-16)
N_and_Nchi_ts_3, C_chi_3, tlist = two_ocsi_sys_3.Average_N_decomposed_evolution(0,10000,1)

"""

"""
#c_list = [5,4]
#c_list = [3+4j,4]
#c_list_1 = [0,10]
c_list_1 = [1,np.sqrt(99)] # <N> starts with 1
#c_list_2 = [5*np.sqrt(2),0]
c_list_2 = [np.sqrt(99/2.),1] # <N> starts with 49.5
c_list_3 = [np.sqrt(100/3),np.sqrt(100/3)] # <N> starts with 33.3
two_ocsi_sys_1 = QuantumOscillators.Chi_analysis(omega_list, c_list_1, 200, 1e-16)
two_ocsi_sys_2 = QuantumOscillators.Chi_analysis(omega_list, c_list_2, 200, 1e-16)
two_ocsi_sys_3 = QuantumOscillators.Chi_analysis(omega_list, c_list_3, 200, 1e-16)
N_and_Nchi_ts_1, C_chi_1, tlist = two_ocsi_sys_1.Average_N_decomposed_evolution(0,1000,1)
N_and_Nchi_ts_2, C_chi_2, tlist = two_ocsi_sys_2.Average_N_decomposed_evolution(0,1000,1)
N_and_Nchi_ts_3, C_chi_3, tlist = two_ocsi_sys_3.Average_N_decomposed_evolution(0,1000,1)
plt.plot(tlist,N_and_Nchi_ts_1[:,0],label='1')
plt.plot(tlist,N_and_Nchi_ts_2[:,0],label='2')
plt.plot(tlist,N_and_Nchi_ts_3[:,0],label='3')
plt.legend()
plt.show()

#two_ocsi_sys = QuantumOscillators.Chi_analysis(omega_list, c_list, 200, 1e-16)

plots_2d_in_3d(tlist,C_chi_3) # This tells you a subsystem takes constant proportion during evolution.
plots_2d_in_3d(tlist,N_and_Nchi_ts_3)
coefficients_chi_decomposed = two_ocsi_sys_3.coefficients_chi_decomposed_evolution(45, tlist)
plots_2d_in_3d(tlist,coefficients_chi_decomposed) # This tells you C_{\chi} is constant.

dm_array, N_averg = two_ocsi_sys_3.density_matrix_evolution(0,1000,1, 'N')
animation_imshow(np.absolute(dm_array))

plt.plot(tlist,N_averg,label='from density matrix')
plt.plot(tlist,N_and_Nchi_ts_3[:,0],label='from Chi synthesis')
plt.legend()
"""