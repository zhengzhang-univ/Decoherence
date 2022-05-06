import numpy as np
from RadiationField import QuantumOscillators
import h5py
import matplotlib.pyplot as plt

omega_list = list(np.array([2e6, 1e6],dtype=int))
c_list = [4,4]
test_cls = QuantumOscillators.Chi_analysis(omega_list, c_list, 94, 1e-16)
test_cls(93,0,100,0.1)

