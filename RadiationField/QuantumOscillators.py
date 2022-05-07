import copy
import math
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from sympy.physics.quantum.constants import hbar
from scipy import linalg
import scipy.special
from . import mpiutil
from . import CoherentState
import h5py
coher_osci_coeff = CoherentState.coher_osci_coeff

def TransferMatrix_rowN(Chi, N):
    return math.sqrt((Chi - 2 * N - 1) * (Chi - 2 * N) * (N + 1))

def solve_Chi_eigen_sys(Chi):
    Nmax = math.floor(Chi / 2)
    A = np.zeros((Nmax + 1, Nmax + 1))
    for i in range(Nmax):
        A[i, i + 1] = A[i + 1, i] = TransferMatrix_rowN(Chi, i + 1)
    eig_vals, eig_vecs = np.linalg.eigh(A)
    return eig_vals, eig_vecs

def solve_Chi_eigen_sys_2(Chi):
    Nmax = math.floor(Chi / 2)
    A = np.zeros((Nmax + 1, Nmax + 1))
    for i in range(Nmax):
        A[i, i + 1] = A[i + 1, i] = TransferMatrix_rowN(Chi, i + 1)
    eig_vals, eig_vecs = np.linalg.eigh(A)
    with h5py.File("Eigenvalues.hdf5", "w") as f:
        f.create_dataset(str(Chi), data=eig_vals)
    with h5py.File("Eigenvectors.hdf5", "w") as f:
        f.create_dataset(str(Chi), data=eig_vecs)
    return None

def solve_whole_system_and_save_2(chimax):
    chi_array = list(np.arange(chimax + 1))
    mpiutil.parallel_jobs_no_gather(solve_Chi_eigen_sys, chi_array, method="alt")
    return None

def solve_whole_system_and_save(chimax):
    chi_array = list(np.arange(chimax + 1))
    Result = mpiutil.parallel_map(solve_Chi_eigen_sys, chi_array, method="alt")
    eig_vals_list, eig_vecs_list = list(zip(*Result))
    if mpiutil.rank0:
        with h5py.File("EigenDecomposition.hdf5","w") as f:
            f.create_dataset("Eigenvalues", data=eig_vals_list, chunks=True)
            f.create_dataset("Eigenvectors", data=eig_vecs_list, chunks=True)
    return

class two_osci_basic():
    def __init__(self, omega_list, c_list, Chimax, Lambda):
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
        self.indices_lists = [[(N, Chi - 2 * N) for N in range(math.floor(Chi / 2) + 1)] for Chi in range(Chimax + 1)]
        self.init_coeff_lists = [np.array([self.get_init_coeff(N, Chi - 2 * N)
                                         for N in range(math.floor(Chi / 2) + 1)])
                               for Chi in range(Chimax + 1)]
        self.filter_chis()
        self.solve_the_system()

    def get_init_coeff(self, N, n):
        return coher_osci_coeff(self.c_list[0], N) * coher_osci_coeff(self.c_list[1], n)

    def filter_chis(self):
        Chimax_old = copy.deepcopy(self.Chimax)
        for i in range(Chimax_old+1):
            chi = Chimax_old-i
            if np.linalg.norm(self.init_coeff_lists[chi]) == 0:
                self.Chimax = chi - 1
            else:
                break
        pass

    def Nmax(self,Chi):
        return math.floor(Chi / 2)

    def solve_the_system(self):
        def aux_crea_f(Chi, N):
            return math.sqrt((Chi - 2 * N + 2) * (Chi - 2 * N + 1) * N)

        def aux_anni_g(Chi, N):
            return math.sqrt((Chi - 2 * N - 1) * (Chi - 2 * N) * (N + 1))

        def solve_Chi_eigen_sys(Chi):
            Nmax = self.Nmax(Chi)
            A = np.zeros((Nmax + 1, Nmax + 1))
            for i in range(Nmax):
                A[i, i + 1] = aux_anni_g(Chi, i)
                A[i + 1, i] = aux_crea_f(Chi, i + 1)
            eig_vals, eig_vecs = scipy.linalg.eigh(A)
            V_H = np.matrix(eig_vecs).getH()
            g_i = np.array(V_H) @ self.init_coeff_lists[Chi]
            return eig_vals, eig_vecs, g_i

        Chi_array = list(np.arange(self.Chimax + 1))
        Result = mpiutil.parallel_map(solve_Chi_eigen_sys, Chi_array, method="alt")
        self.eig_vals_lists, self.eig_vecs_lists, self.init_cond_lists = list(zip(*Result))
        #self.scaled_eig_vals = [self.factor*item for item in self.eig_vals_lists]
        #self.scaled_eig_vecs = [np.einsum("ij,j->ij",self.eig_vecs_lists[i],self.init_cond_lists[i])
        #                        for i in range(self.Chimax + 1)]

    def coeffs_normalized_t(self, t):
        tt = self.factor * t
        Chi_array = list(np.arange(self.Chimax + 1))
        def linear_solver(Chi):
            basis = np.exp(self.eig_vals_lists[Chi] * tt)
            aux = np.einsum("ij, j, j -> i", self.eig_vecs_lists[Chi], self.init_cond_lists[Chi], basis)
            module = np.linalg.norm(aux) ** 2
            return aux, module
        Result = mpiutil.parallel_map(linear_solver, Chi_array, method="alt")
        coeffs,amp2= list(zip(*Result))
        norm = math.sqrt(sum(amp2))
        return [item/norm for item in coeffs]

    def linear_solver_chi_basis(self, Chi: int):
        basis = np.exp(self.factor * self.eig_vals_lists[Chi] * self.t)
        coeffs_chi_Ns_t = np.einsum("ij, j, j -> i", self.eig_vecs_lists[Chi], self.init_cond_lists[Chi], basis)
        coeffs_chi_t = np.linalg.norm(coeffs_chi_Ns_t) ** 2
        Nmax = self.Nmax(Chi)
        N_averg_chi = sum( np.arange(Nmax+1) * np.absolute(coeffs_chi_Ns_t**2) )/ coeffs_chi_t
        return coeffs_chi_t, N_averg_chi

    def Observable_N_t(self, t):
        self.t = t
        Chi_array = list(np.arange(self.Chimax + 1))
        Result = mpiutil.parallel_map(self.linear_solver_chi_basis, Chi_array, method="alt")
        c_chi, N_averg_chi= list(zip(*Result))
        return sum(np.array(N_averg_chi)*np.array(c_chi))/sum(np.array(c_chi))

    def evolution_of_coefficient_magnitude(self,N,n,t):
        tt = self.factor * t
        Chi = 2*N + n
        basis = np.exp(self.eig_vals_lists[Chi] * tt)
        result = np.einsum("j, j, j", self.eig_vecs_lists[Chi][N,:], self.init_cond_lists[Chi], basis)
        return np.absolute(result)

    def turn_list_to_array(self, lists):
        result = np.zeros((self.Chimax + 1, self.Chimax + 1), dtype=complex)
        for i in range(self.Chimax + 1):
            for j in range(math.floor(i / 2) + 1):
                a, b = self.indices_lists[i][j]
                result[a, b] = lists[i][j]
        return result

    def from_list_to_density_mat(self, lists, oscillator):
        C_Nn = self.turn_list_to_array(lists)
        result = None
        if oscillator == 'n':
            result = np.einsum("ij,ik->jk", C_Nn.conjugate(), C_Nn)
        elif oscillator == 'N':
            result = np.einsum("ji,ki->jk", C_Nn.conjugate(), C_Nn)
        photons = np.sum(result.diagonal() * np.arange(self.Chimax + 1)).real
        return result, photons

    def density_matrix_t(self, t, oscillator):
        clist = self.coeffs_normalized_t(t)
        return self.from_list_to_density_mat(clist,oscillator)

    def density_matrix_evolution(self, st, et, dt, oscillator):
        tlist = np.arange(st, et, dt)
        Result = [self.density_matrix_t(t,oscillator) for t in tlist]
        density_matrices, num_photons = list(zip(*Result))
        return np.array(density_matrices), np.array(num_photons)

class two_osci_solved(two_osci_basic):
    def solve_the_system(self):
        eigen_data = h5py.File("EigenDecomposition.hdf5", 'r')
        self.eig_vals_lists=eigen_data["Eigenvalues"]
        self.eig_vecs_lists=eigen_data["Eigenvectors"]
        def get_initial_conditions(chi):
            V_H = np.matrix(self.eig_vecs_lists[chi]).getH()
            g_i = np.array(V_H) @ self.init_coeff_lists[chi]
            return g_i
        Chi_array = list(np.arange(self.Chimax + 1))
        self.init_cond_lists = mpiutil.parallel_map(get_initial_conditions, Chi_array, method="alt")
        return

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
        return [self.single_coefficient_t(Chi, N, t) for t in ts]

    def single_coefficient_t(self, Chi, N, t):
        """
        Return: the square of the amplitude.
        """
        tt = self.factor * t
        basis = np.exp(self.eig_vals_lists[Chi] * tt)
        result = np.einsum("j, j, j", self.eig_vecs_lists[Chi][N, :], self.init_cond_lists[Chi], basis)
        return np.absolute(result) ** 2

    def coefficients_chi_decomposed_evolution(self, Chi, ts):
        return np.array([self.coefficients_chi_decomposed_t(Chi, t) for t in ts])

    def coefficients_chi_decomposed_t(self, Chi, t):
        tt = self.factor * t
        basis = np.exp(self.eig_vals_lists[Chi] * tt)
        aux = np.einsum("ij, j, j->i", self.eig_vecs_lists[Chi], self.init_cond_lists[Chi], basis)
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


