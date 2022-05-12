import math
import numpy as np
#import matplotlib.pyplot as plt
import sympy
from sympy.physics.quantum.constants import hbar
from sympy.physics.qho_1d import coherent_state
from . import mpiutil
import h5py

class two_osci_solved():
    def __init__(self, omega_list, c_list, Chimax, Lambda, path):
        """
        c_list = [C, c]
        m_list = [M, m]
        """
        self.transfer_matrices = None
        hb = sympy.N(hbar)
        m_list = [hb * omega for omega in omega_list]
        self.factor = float(
            Lambda * math.sqrt(hb / (2 * m_list[0] * omega_list[0])) * (hb / (2 * m_list[1] * omega_list[1])) / hb) * (
                          -1j)
        self.Chimax = Chimax
        self.c_list = np.array(c_list)
        self.datapath = path
        self.f1 = h5py.File(path + "eigenvalues.hdf5", 'r')
        self.f2 = h5py.File(path + "eigenvectors.hdf5", 'r')
        self.indices_lists = [[(N, Chi - 2 * N) for N in range(math.floor(Chi / 2) + 1)]
                               for Chi in range(Chimax + 1)]
        self.init_coeff_lists = [sympy.Matrix([self.get_init_coeff(N, Chi - 2 * N)
                                               for N in range(math.floor(Chi / 2) + 1)])
                                 for Chi in range(Chimax + 1)]
        self.solve_initial_conditions()
        if mpiutil.rank0:
            self.create_auxiliary_array()
        mpiutil.barrier()
        self.projected_vecs_f = h5py.File(path+'aux_array.hdf5', 'r')

    def get_init_coeff(self, N, n):
        return coherent_state(N,self.c_list[0])*coherent_state(n,self.c_list[1])

    def Nmax(self,Chi):
        return math.floor(Chi / 2)

    def solve_initial_conditions(self):
        def get_initial_condition(chi):
            V = sympy.Matrix(self.eigen_vecs(chi))
            g_i =  V.H * self.init_coeff_lists[chi]
            return sympy.N(g_i) #type sympy Matrix
        Chi_array = list(np.arange(self.Chimax + 1))
        self.init_cond_lists = mpiutil.parallel_map(get_initial_condition, Chi_array, method="alt")
        return


    def create_auxiliary_array(self):
        f = h5py.File(self.datapath+'aux_array.hdf5','w')
        for chi in range(self.Chimax+1):
            aux_array = self.eigen_vecs(chi) @ np.array(sympy.diag(*list(self.init_cond_lists[chi])))
            dset = f.create_dataset('{0}'.format(chi), aux_array.shape)
            dset[:,:]=aux_array.astype(complex)
        f.close()
        print("Eigenvector array has been projected!")


    def create_auxiliary_array_parallel(self):
        rank = mpiutil.rank
        size = mpiutil.size
        nbatch = math.floor(self.Chimax / size)
        f = h5py.File('aux_array.hdf5','w', driver='mpio', comm=mpiutil._comm)
        dset = []
        for chi in range(self.Chimax+1):
            Nmax = self.Nmax(chi)
            dset.append(f.create_dataset('{0}'.format(chi),(Nmax+1,Nmax+1)))
        mpiutil.barrier()
        for i in range(nbatch):
            chi = i * size+rank
            aux_array = self.eigen_vecs(chi)@sympy.diag(*list(self.init_cond_lists[chi]))
            dset[chi][:,:] = np.array(aux_array).astype(complex)
        chi = nbatch * size + rank
        if chi <= self.Chimax:
            aux_array = self.eigen_vecs(chi)@sympy.diag(*list(self.init_cond_lists[chi]))
            dset[chi][:,:] = np.array(aux_array).astype(complex)
        f.close()
        self.projected_vecs_f = h5py.File('aux_array.hdf5','r')


    def eigen_vals(self, chi):
        return self.f1[str(chi)][...]

    def eigen_vecs(self, chi):
        return self.f2[str(chi)][...]

    def projected_vecs(self, chi):
        return self.projected_vecs_f[str(chi)][...]

    def all_coeffs_normalized_t(self, t):
        tt = self.factor * t
        Chi_array = list(np.arange(self.Chimax + 1))
        def linear_solver(Chi):
            basis = (np.exp(self.eigen_vals(Chi) * tt)).reshape(-1,1)
            aux = self.projected_vecs(Chi) * sympy.Matrix(basis)
            return aux
        result = mpiutil.parallel_map(linear_solver, Chi_array, method="alt")
        return result
    """
    def all_coeffs_normalized_t(self, t):
        tt = self.factor * t
        Chi_array = list(np.arange(self.Chimax + 1))
        def linear_solver(Chi):
            basis = (np.exp(self.eigen_vals(Chi) * tt)).reshape(-1,1)
            aux_array = sympy.Matrix(self.projected_vecs(Chi))*sympy.diag(*list(self.init_cond_lists[Chi]))
            aux = aux_array * sympy.Matrix(basis)
            #amp_square = aux.norm()**2
            return aux #, amp_square
        result = mpiutil.parallel_map(linear_solver, Chi_array, method="alt")
        #coeffs,amp2= list(zip(*Result))
        #norm = sympy.sqrt(sum(amp2))
        #return [item/norm for item in coeffs]
        return result
    """
    def N_avrg(self, t):
        tt = self.factor * t
        Chi_array = list(np.arange(self.Chimax + 1))

    def turn_list_to_array(self, lists):
        result = sympy.zeros(self.Nmax(self.Chimax) + 1, self.Chimax + 1)
        for i in range(self.Chimax + 1):
            for j in range(math.floor(i / 2) + 1):
                a, b = self.indices_lists[i][j]
                result[a, b] = lists[i][j]
        return result

    def from_list_to_density_mat(self, lists, oscillator):
        C_Nn = self.turn_list_to_array(lists)
        A,b=sympy.shape(C_Nn)
        result = None
        photon_num_avrg = None
        if oscillator == 'n':
            result = C_Nn.H * C_Nn
            photon_num_avrg = float(sum([result[i,i]*i for i in range(b)]))
        elif oscillator == 'N':
            result = C_Nn * C_Nn.H
            photon_num_avrg = float(sum([result[i,i]*i for i in range(A)]))
        return result, photon_num_avrg

    def density_matrix_t(self, t, oscillator):
        clist = self.all_coeffs_normalized_t(t)
        return self.from_list_to_density_mat(clist,oscillator)

    def density_matrix_evolution(self, st, et, dt, oscillator):
        tlist = np.arange(st, et, dt)
        result = [self.density_matrix_t(t,oscillator) for t in tlist]
        density_matrices, num_photons = list(zip(*result))
        return np.array(density_matrices,dtype=complex), np.array(num_photons, dtype=float)



