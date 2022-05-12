from sympy import *
import numpy as np
import math as M
import scipy.special
from sympy.physics.qho_1d import psi_n
from sympy.physics.quantum.constants import hbar
import matplotlib.pyplot as plt
from matplotlib import cm
import copy

x, xa, xb, t = symbols("x x_a x_b t", real = True)

class n_coherent_oscillators():
    def __init__(self, symbol_list, omega_list, c_list):
        """
        symbol_list: symbols for the degrees of freedom, i.e., displacement variables of oscillators.
        omega_list: array(n,) ---  Frequency quanta of the oscillators.
        c_list: array(n,) ---  Coherents state parameters for the oscillators.
        """
        self.alphabets = np.array(['a','b','c','d','e','f','g','h','i','j','k', 
                                   'l','m','n','o','p','q','r','s','t','u','v', 
                                   'w','x','y','z'])
        self.symbs = symbol_list
        self.omegas = np.array(omega_list)
        self.coher_params = np.array(c_list)
        self.masses = 2*hbar*self.omegas
        #assert len(omega_list) == len(c_list)
        self.num_of_osci = len(c_list)
        self.max_Fock_index = int(10 * np.rint(np.max(np.abs(self.coher_params))))
        self.init_coeff_matrix = self.n_coher_osci_coeffs(self.coher_params)
        init_coeffs = [array for array in self.init_coeff_matrix]
        self.init_coeff_array = self.expand_coeff_matrix(init_coeffs)
        oper_array = np.zeros((self.max_Fock_index + 4, self.max_Fock_index + 4))
        for i in range(self.max_Fock_index + 3):
            oper_array[i, i+1] = M.sqrt(i+1)
        self.single_operater = oper_array
        self.anni_matrix = oper_array
        self.anni_matrix_2 = oper_array @ oper_array
        self.crea_matrix = oper_array.T
        self.crea_matrix_2 = (oper_array.T)@(oper_array.T)
        self.lam = N(hbar*10**18)
        self.factor = 2*M.sqrt(2)*self.lam/(8j*N(hbar)*self.omegas[0]**2*self.omegas[1])

        
        
    def single_coher_osci_coeffs(self, c, n):
        """
        c: the coherent state parameter
        n: the index/indices of the Fock state(s). Could be a single interger or an interger array.
        """
        coherent_coeffs = np.exp(-0.5*np.abs(c)**2)* c ** n / np.sqrt(scipy.special.factorial(n))
        return np.array(coherent_coeffs, dtype=complex)
        
    def n_coher_osci_coeffs(self, cs):
        num = self.num_of_osci
        length = self.max_Fock_index + 4
        result = []
        #result = np.empty([num, length], dtype=complex)
        Fock_indices = np.arange(length)
        for i in np.arange(num):
            result.append( self.single_coher_osci_coeffs(cs[i], Fock_indices))
        return result
    
    def coeff_evol(self, t, y):
        Coeffs = np.reshape(y,(self.max_Fock_index+4,self.max_Fock_index+4)) 
        dydt = (self.crea_matrix_2 @ Coeffs @ self.crea_matrix
                + self.anni_matrix_2 @ Coeffs @ self.anni_matrix) * self.factor
        return dydt.flatten()
    
    def coeff_evol_old(self, t, y):
        Coeffs = np.reshape(y,(self.max_Fock_index+4,self.max_Fock_index+4)) 
        dydt = (self.apply_operators(Coeffs , (0,'+'), (0,'+'), (1,'-')) 
                + self.apply_operators(Coeffs , (0,'-'), (0,'-'), (1,'+'))) * self.factor
        return dydt.flatten()
    
    
    def apply_operators(self, coeff_array, *ind):
        """
        input: (oscillator index, operator type) pairs
        ------------
            Example: apply_operators( (0,"-"),(0,"-"),(1,"+"))
            
        return: 
        
        """
        result = copy.deepcopy(coeff_array)
        for pair in ind:
            result = self.apply_single_oper(result, pair[0], pair[1])
        """
        for i in range(self.num_of_osci):
            aux = copy.deepcopy(result)
            del result
            result = np.delete(aux, [-1,-2,-3,-4], axis = i)
        """
        return result

    
    def apply_single_oper(self, coeffs_array, ind, kind):
        """
        "ijk,"
        """
        alphabets = copy.deepcopy(self.alphabets)
        indices = alphabets[:self.num_of_osci]
        string1 = "".join(indices)
        indices[ind] = self.alphabets[self.num_of_osci]
        string2 = "".join(indices)
        if kind == "-":
            string = string1 + "," + self.alphabets[self.num_of_osci] + self.alphabets[int(ind)] + "->" + string2
        elif kind == "+":
            string = string1 + "," + self.alphabets[int(ind)] + self.alphabets[self.num_of_osci] + "->" + string2
        else:
            raise NameError('Invalid name for operators. Should be either \'+\' or \'-\'. ')
        result = np.einsum(string, coeffs_array, self.single_operater)
        return result

    
    def expand_coeff_matrix(self, coeff_list):
        aux_string=''
        for i in range(self.num_of_osci):
            aux_string+= self.alphabets[i]+','
        iterable = [aux_string[:-1]] + list(coeff_list)
        return np.einsum(*iterable)
        
    def evolution_with_interaction(self, Lambda_aab):
        t = symbols("t",real=True)
        Factor = -Lambda_aab*t*1j/(8*hbar*self.omegas[0]*self.omegas[0]*self.omegas[1])
        aux1 = self.apply_operators(self.init_coeff_array, (0,'-'), (0,'-'), (1,'+'))
        aux2 = self.apply_operators(self.init_coeff_array, (0,'+'), (0,'+'), (1,'-'))
        #init_coeffs = [array[:self.max_Fock_index] for array in self.init_coeff_matrix]
        #outer_1 = self.expand_coeff_matrix(aux1)
        #outer_2 = self.expand_coeff_matrix(aux2)
        #outer_i = self.expand_coeff_matrix(init_coeffs)
        initial = self.init_coeff_array
        for i in range(self.num_of_osci):
            aux = copy.deepcopy(initial)
            del initial
            initial = np.delete(aux, [-1,-2,-3,-4], axis = i)
        result = (aux1 + aux2) * Factor + initial 
        return result
    
    def density_matrix(self, ind, coeff_array, time):
        """
        ind: integer. The index of the considered oscillator.
        time: real positive. Evolution time.
        coeff_array: unseparable expression for 
        """
        indices = copy.deepcopy(self.alphabets[:self.num_of_osci])
        aux_string_1 = "".join(indices)
        indices[ind] = self.alphabets[self.num_of_osci]
        aux_string_2 = "".join(indices)
        aux_string = aux_string_1 + ',' + aux_string_2 + '->' + aux_string_1[ind] + aux_string_2[ind]
        aux = np.array(N(Matrix(coeff_array).subs(t,time)),dtype=complex)
        aux_n = aux/np.linalg.norm(aux)
        return np.einsum(aux_string, np.conjugate(aux_n), aux_n)

from scipy.integrate import solve_ivp

symbol_list = [xa,xb]
omega_list = np.array([1e6,2e6],dtype=int)
c_list = [4,4]
test = n_coherent_oscillators(symbol_list, omega_list, c_list)
y0 = test.init_coeff_array.flatten()
#time = np.linspace(0, 50, 10)

sol = solve_ivp(test.coeff_evol, (0,1000), y0)

import h5py

with h5py.File("coeffs_old_way.hdf5", "w") as f:
    f.create_dataset("t",data=sol['t'])
    f.create_dataset("y",data=sol['y'])

