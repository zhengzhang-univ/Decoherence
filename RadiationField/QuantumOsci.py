import math
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from sympy.physics.quantum.constants import hbar
from . import mpiutil
from . import CoherentState
coher_osci_amp = CoherentState.coher_osci_amp


class two_modes_coupled():
    def __init__(self, omega_list, c_list, Chimax, Lambda):
        """
        c_list = [C, c]
        m_list = [M, m]
        """
        hb = sp.N(hbar)
        m_list = [hb * omega for omega in omega_list]
        self.factor = float(Lambda * math.sqrt(hb / (2 * m_list[0] * omega_list[0])) * (hb / (2 * m_list[1] * omega_list[1])) / hb) * (-1j)
        self.Chimax = Chimax
        self.c_list = np.array(c_list)
        self.indices_lists = [[(N, Chi - 2 * N) for N in range(math.floor(Chi / 2) + 1)] for Chi in range(Chimax + 1)]
        self.init_amp_lists = [sp.Matrix([self.get_init_amp(N, Chi - 2 * N) for N in range(math.floor(Chi / 2) + 1)]) for Chi in
                               range(Chimax + 1)]
        self.solve_the_system()

    def get_init_amp(self, N, n):
        return coher_osci_amp(self.c_list[0], N) * coher_osci_amp(self.c_list[1], n)

    def solve_the_system(self):
        def aux_crea_f(Chi, j):
            return sp.sqrt((Chi - 2 * j + 2) * (Chi - 2 * j + 1) * j)

        def aux_anni_g(Chi, j):
            return sp.sqrt((Chi - 2 * j - 1) * (Chi - 2 * j) * (j + 1))

        def solve_Chi_eigen_sys(Chi):
            Nmax = math.floor(Chi / 2)
            A = sp.zeros(Nmax + 1,Nmax + 1)
            for i in range(Nmax):
                A[i, i + 1] = aux_anni_g(Chi, i)
                A[i + 1, i] = aux_crea_f(Chi, i + 1)
            EigenDecomposition = A.eigenvects() # list object of tuples: [(eigenvalues, degeneracy, eigenvectors), ...]
            eig_vals = sp.zeros(Nmax + 1, 1)
            eig_vects = sp.zeros(Nmax + 1,Nmax + 1)
            aux = 0
            for item in EigenDecomposition:
                for j in np.arange(item[1]):
                    eig_vects[:, aux + j] = item[2][j]
                    eig_vals[aux+j] = item[0]
                aux += item[1]
            V_H = eig_vects.H
            c_i = V_H  @ self.init_amp_lists[Chi]
            return np.array(sp.N(eig_vals)), np.array(sp.N(eig_vects)), np.array(sp.N(c_i))

        Chi_array = list(np.arange(self.Chimax + 1))
        Result = mpiutil.parallel_map(solve_Chi_eigen_sys, Chi_array, method="alt")
        self.eig_vals_lists, self.eig_vecs_lists, self.init_cond_lists = list(zip(*Result))

    def time_evolution(self, t):
        Chi_array = list(np.arange(self.Chimax + 1))
        tt = self.factor * t
        def linear_solver(Chi):
            e_factor = np.array([np.exp(self.eig_vals_lists[i] * tt) * self.init_cond_lists[Chi]
                                 for i in range(Chi + 1)]).astype(complex)
            aux = np.array(self.eig_vecs_lists[Chi] @ e_factor).astype(complex)
            module = np.linalg.norm(aux) ** 2
            return aux, module

        Result = mpiutil.parallel_map(linear_solver, Chi_array, method="alt")
        coeffs,amp2= list(zip(*Result))
        return np.array(coeffs) / math.sqrt(sum(amp2))

    def turn_list_to_array(self, lists):
        result = np.zeros((self.Chimax + 1, self.Chimax + 1), dtype=complex)
        for i in range(self.Chimax + 1):
            for j in range(math.floor(i / 2) + 1):
                a, b = self.indices_lists[i][j]
                result[a, b] = lists[i][j]
        return result

    def density_mat(self, lists, osci_ind):
        C_Nn = self.turn_list_to_array(lists)
        result = None
        if osci_ind == 'n':
            result = np.einsum("ij,ik->jk", C_Nn.conjugate(), C_Nn)
        elif osci_ind == 'N':
            result = np.einsum("ji,ki->jk", C_Nn.conjugate(), C_Nn)
        E = np.sum(result.diagonal() * np.arange(self.Chimax + 1)).real + 0.5
        return result, E

    def talked_modes_analysis(self, Chi, ts):
        modes_num = len(self.indices_lists[Chi])
        times_num = len(ts)
        result = np.zeros((modes_num, times_num), dtype='complex')
        for i in range(times_num):
            result[:, i] = self.time_evolution(ts[i])[Chi]
        fig, (ax1, ax2) = plt.subplots(2, 1)
        fig.set_size_inches(10.5, 6.5)

        fig.suptitle('Title of figure')

        ax1.set_title('Amplitude')
        im1 = ax1.imshow(np.absolute(result), cmap='gray')

        ax2.set_title('Phase')
        im2 = ax2.imshow(np.angle(result), cmap='gray')
        # plt.imshow(result, cmap='gray')
        plt.tight_layout()
        plt.show()
