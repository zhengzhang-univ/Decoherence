import numpy as np

from RadiationField.MixingMatricesSolver import parallel_solve_mixing_matrices_and_save

def get_total_nbytes(chimax):
    return np.sum(4*2*np.arange(np.floor(chimax/2)+1)**2)
#parallel_solve_mixing_matrices_and_save(0,4000)
path = "/data/zzhang/eigendata/"
parallel_solve_mixing_matrices_and_save(0,10000,path)