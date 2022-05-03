import numpy as np
from RadiationField import QuantumOsci

omega_list = list(np.array([2e6, 1e6],dtype=int))
c_list = [4,4]
test_cls = QuantumOsci.two_modes_coupled(omega_list, c_list, 100, 1e-16)
times = np.arange(400)*.25