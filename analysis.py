import matplotlib.pyplot as plt
import numpy as np
import h5py

path = "/Users/zheng/Dropbox/project with Nick/"
f=h5py.File(path+"N_averg.hdf5",'r')

N1 = f['N_sys1'][...]
N2 = f['N_sys2'][...]
N3 = f['N_sys3'][...]
N4 = f['N_sys4'][...]

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
c_list_1 = [0,np.sqrt(3999/2.)]
c_list_2 = get_c_list(chimax, 30)
c_list_3 = get_c_list(chimax, 15)
c_list_4 = get_c_list(chimax, 0)

plt.plot(N1)
plt.plot(N2)
plt.plot(N3)
plt.plot(N4)

plt.show()
