import math
import numpy as np
import sympy
from sympy.physics.quantum.constants import hbar
from . import mpiutil
import h5py
from RadiationField.CoherentState import log_coeff

class two_osci_solved():
    def __init__(self, omega_list, c_list, Chimax, Lambda, path):
        hb = sympy.N(hbar)
        m_list = [hb * omega for omega in omega_list]
        self.factor = float(Lambda * math.sqrt(hb / (2 * m_list[0] * omega_list[0])) *
                            (hb / (2 * m_list[1] * omega_list[1])) / hb
                            ) * (-1j)
        self.Chimax = Chimax
        self.log_scaling = 0
        self.c_list = np.array(c_list)
        self.datapath = path
        f1 = h5py.File(path + "eigenvalues.hdf5", 'r')
        f2 = h5py.File(path + "eigenvectors.hdf5", 'r')
        self.indices_lists = [[(N, Chi - 2 * N) for N in range(math.floor(Chi / 2) + 1)]
                               for Chi in range(Chimax + 1)]

        self.local_chis = mpiutil.partition_list_mpi(np.arange(Chimax+1), method="alt", comm=mpiutil._comm)
        # self.init_coeff_lists = mpiutil.parallel_map(get_init_coeff_chi, list(np.arange(Chimax + 1)))

        def eigen_vals(chi):
            return f1[str(chi)][...]

        def eigen_vecs(chi):
            return f2[str(chi)][...]

        self.eig_vals = [eigen_vals(chi) for chi in self.local_chis]
        self.eig_vecs = [eigen_vecs(chi) for chi in self.local_chis]
        mpiutil.barrier()
        f1.close()
        f2.close()
        self.local_scaled_init_log_coeff_lists = [self.get_init_log_coeff_chi(chi) for chi in self.local_chis]
        mpiutil.barrier()
        self.solve_initial_conditions()

    def get_init_log_coeff(self, N, n):
        return log_coeff(self.c_list[0],N) + log_coeff(self.c_list[1],n) + self.log_scaling

    def get_init_log_coeff_chi(self,Chi):
        result = np.array([self.get_init_log_coeff(N, Chi - 2 * N)
                           for N in range(self.Nmax(Chi) + 1)])
        return result

    def Nmax(self,Chi):
        return math.floor(Chi / 2)

    def solve_initial_conditions(self):
        def get_initial_condition(ind):
            V = np.matrix(self.eig_vecs[ind])
            g_i = np.einsum("ij,j -> i", np.array(V.H), np.exp(self.local_scaled_init_log_coeff_lists[ind]))
            flag = 1
            if np.sum(g_i) == 0:
                flag = 0
            return g_i, flag
        # Chi_array = list(np.arange(self.Chimax + 1))
        # self.init_cond_lists = mpiutil.parallel_map(get_initial_condition, Chi_array, method="alt")
        Result = [get_initial_condition(ind) for ind in range(len(self.local_chis))]
        self.local_init_cond_lists, self.flags = list(zip(*Result))
        if mpiutil.rank0:
            print('The system has been initialized!')

    def all_coeffs_t(self, t):
        tt = self.factor * t
        Chi_array = list(np.arange(self.Chimax + 1))
        def linear_solver(Chi):
            ind = math.floor(Chi / mpiutil.size)
            if self.flags[ind] == 0:
                aux = np.zeros(self.Nmax(Chi),dtype=complex)
                return aux, 0
            else:
                Nmax = self.Nmax(Chi)
                basis = np.exp(self.eig_vals[ind] * tt)
                aux = np.einsum("ij, j, j -> i", self.eig_vecs[ind], self.local_init_cond_lists[ind], basis)
                N_avrg = np.exp(-2*self.log_scaling) * sum(np.absolute(aux) ** 2 * np.arange(Nmax+1))
                return aux, N_avrg
        result = mpiutil.parallel_map(linear_solver, Chi_array, method="alt")
        coeffs, N_avrg_decomp = list(zip(*result))
        N_avrg = sum(N_avrg_decomp)
        return coeffs, N_avrg

    def turn_list_to_array(self, lists):
        result = np.zeros((self.Nmax(self.Chimax) + 1, self.Chimax + 1), dtype=complex)
        for i in range(self.Chimax + 1):
            for j in range(math.floor(i / 2) + 1):
                a, b = self.indices_lists[i][j]
                result[a, b] = lists[i][j]
        return result

    def from_list_to_density_mat(self, lists, oscillator):
        C_Nn = self.turn_list_to_array(lists)
        #A,b=sympy.shape(C_Nn)
        result = None
        #photon_num_avrg = None
        if oscillator == 'n':
            result = np.array(np.matrix(C_Nn).H) @ C_Nn
            #photon_num_avrg = float(sum([result[i,i]*i for i in range(b)]))
        elif oscillator == 'N':
            result = C_Nn @ np.array(np.matrix(C_Nn).H)
            #photon_num_avrg = float(sum([result[i,i]*i for i in range(A)]))
        return result #, photon_num_avrg

    def density_matrix_t(self, t, oscillator):
        clist, aux = self.all_coeffs_t(t)
        return self.from_list_to_density_mat(clist,oscillator)

    def density_matrix_evolution(self, st, et, dt, oscillator):
        tlist = np.arange(st, et, dt)
        dm_t=[]
        N_t=[]
        for t in tlist:
            coeff_list, N_avrg = self.all_coeffs_t(t)
            dm_t.append(self.from_list_to_density_mat(coeff_list,oscillator))
            N_t.append(N_avrg)
        return np.array(dm_t,dtype=complex), np.array(N_t, dtype=float)

    def N_averg_evolution(self,st, et, dt):
        tlist = np.arange(st, et, dt)
        N_t=[]
        for t in tlist:
            coeff_list, N_avrg = self.all_coeffs_t(t)
            N_t.append(N_avrg)
        return np.array(N_t)