import math
import numpy as np
import matplotlib.pyplot as plt
import sympy
import sympy as sp
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
        hb = sp.N(hbar)
        m_list = [hb * omega for omega in omega_list]
        self.factor = float(
            Lambda * math.sqrt(hb / (2 * m_list[0] * omega_list[0])) * (hb / (2 * m_list[1] * omega_list[1])) / hb) * (
                          -1j)
        self.Chimax = Chimax
        self.c_list = np.array(c_list)
        self.f1 = h5py.File(path + "eigenvalues.hdf5", 'r')
        self.f2 = h5py.File(path + "eigenvectors.hdf5", 'r')
        self.indices_lists = [[(N, Chi - 2 * N) for N in range(math.floor(Chi / 2) + 1)]
                               for Chi in range(Chimax + 1)]
        self.init_coeff_lists = [sympy.Matrix([self.get_init_coeff(N, Chi - 2 * N)
                                               for N in range(math.floor(Chi / 2) + 1)])
                                 for Chi in range(Chimax + 1)]
        self.solve_initial_conditions()

    def get_init_coeff(self, N, n):
        return coherent_state(N,self.c_list[0])*coherent_state(n,self.c_list[1])

    def Nmax(self,Chi):
        return math.floor(Chi / 2)

    def solve_initial_conditions(self):
        f = h5py.File(self.datapath+"eigenvectors.hdf5", 'r')
        def get_initial_condition(chi):
            V = sympy.Matrix(f[str(chi)][...])
            g_i =  V.H * self.init_coeff_lists[chi]
            return g_i #type sympy Matrix
        Chi_array = list(np.arange(self.Chimax + 1))
        self.init_cond_lists = mpiutil.parallel_map(get_initial_condition, Chi_array, method="alt")
        return

    def eigen_vals(self, chi):
        return self.f1[str(chi)][...]

    def eigen_vecs(self, chi):
        return self.f2[str(chi)][...]

    def all_coeffs_normalized_t(self, t):
        tt = self.factor * t
        Chi_array = list(np.arange(self.Chimax + 1))
        def linear_solver(Chi):
            basis = (np.exp(self.eigen_vals(Chi) * tt)).reshape(-1,1)
            aux_array = sympy.Matrix(self.eigen_vecs(Chi))*sympy.diag(*list(self.init_cond_lists[Chi]))
            aux = aux_array * sympy.Matrix(basis)
            amp_square = aux.norm()**2
            return aux, amp_square
        Result = mpiutil.parallel_map(linear_solver, Chi_array, method="alt")
        coeffs,amp2= list(zip(*Result))
        norm = sympy.sqrt(sum(amp2))
        return [item/norm for item in coeffs]

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
            photon_num_avrg = sympy.N(sum([result[i,i]*i for i in range(b)]))
        elif oscillator == 'N':
            result = C_Nn * C_Nn.H
            photon_num_avrg = sympy.N(sum([result[i,i]*i for i in range(A)]))
        return result, photon_num_avrg

    def density_matrix_t(self, t, oscillator):
        clist = self.all_coeffs_normalized_t(t)
        return self.from_list_to_density_mat(clist,oscillator)

    def density_matrix_evolution(self, st, et, dt, oscillator):
        tlist = np.arange(st, et, dt)
        result = [self.density_matrix_t(t,oscillator) for t in tlist]
        density_matrices, num_photons = list(zip(*result))
        return np.array(density_matrices,dtype=complex), np.array(num_photons, dtype=float)

class Chi_analysis(two_osci_solved):
    def Average_N_decomposed_evolution(self, st, et, dt):
        """
        Return: 2d array (ts,Chimax+2)
        """
        tlist = np.arange(st, et, dt)
        Result = [self.Average_N_decomposed_t(t) for t in tlist]
        N_and_Nchi_ts, Cs_chi = list(zip(*Result))
        return np.array(N_and_Nchi_ts), np.array(Cs_chi), tlist

    def Average_N_decomposed_t(self, t):
        """
        Return: 1d array (Chimax+2,)
        """
        self.t = t
        Chi_array = list(np.arange(self.Chimax + 1))
        Result = mpiutil.parallel_map(self.linear_solver_chi_basis, Chi_array, method="alt")
        c_chi, N_averg_chi= list(zip(*Result))
        C_chi_normed = np.array(c_chi) / sum(np.array(c_chi))
        N_averg_chi_scaled = np.array(N_averg_chi)*C_chi_normed
        N_averg = sum(N_averg_chi_scaled)
        return np.append(N_averg, N_averg_chi_scaled), C_chi_normed

    def single_coefficient_evolution(self, Chi, N, ts):
        """
        Return: Time sequence of the square of the amplitude of the coefficient.
        """
        return [self.single_coefficient_amp_square_t(Chi, N, t) for t in ts]

    def single_coefficient_amp_square_t(self, Chi, N, t):
        """
        Return: the square of the amplitude.
        """
        tt = self.factor * t
        basis = (np.exp(self.eigen_vals(Chi) * tt)).reshape(-1,1)
        aux_array = sympy.Matrix(self.eigen_vecs(Chi)) * sympy.diag(*list(self.init_cond_lists[Chi]))
        aux = aux_array * sympy.Matrix(basis)
        return np.absolute(sympy.N(aux[N])) ** 2

    def coefficients_chi_decomposed_evolution(self, Chi, ts):
        return np.array([self.coefficients_chi_decomposed_t(Chi, t) for t in ts])

    def coefficients_chi_decomposed_t(self, Chi, t):
        tt = self.factor * t
        basis = (np.exp(self.eigen_vals(Chi) * tt)).reshape(-1, 1)
        aux_array = sympy.Matrix(self.eigen_vecs(Chi)) * sympy.diag(*list(self.init_cond_lists[Chi]))
        aux = aux_array * sympy.Matrix(basis)
        result=np.absolute(aux) ** 2
        #result_weighted_by_N = result*np.arange(self.Nmax(Chi)+1)/sum(result)
        c_chi = sum(result)
        return np.append(c_chi,result)

    def coeffs_chi_globally_normalized_gray_scale(self, Chi, st, et, dt):
        ts = np.linspace(st,et,dt)
        modes_num = len(self.indices_lists[Chi])
        times_num = len(ts)
        result = np.zeros((modes_num, times_num), dtype='complex')
        for i in range(times_num):
            result[:, i] = self.coeffs_normalized_t(ts[i])[Chi]
        fig, (ax1, ax2) = plt.subplots(2, 1)
        fig.set_size_inches(10.5, 6.5)

        fig.suptitle('$\chi=$'+str(Chi))

        ax1.set_title('Amplitude')
        im1 = ax1.imshow(np.absolute(result), cmap='gray')

        ax2.set_title('Phase')
        im2 = ax2.imshow(np.angle(result), cmap='gray')
        #plt.imshow(result, cmap='gray')
        plt.tight_layout()
        plt.show()


