from sympy import *
import numpy as np
import math as M
from sympy.physics.qho_1d import psi_n, coherent_state
init_printing()

def log_coeff_list(c, ns: list):
    a = len(list(ns))
    coeffarray = np.zeros(a, dtype=complex)
    aux1 = -0.5 * np.abs(c) ** 2
    aux2 = np.log(c)
    for i in range(a):
        coeffarray[i] = aux1 + ns[i] * aux2 - 0.5 * np.sum(np.log(np.arange(1, ns[i]+1)))
    return coeffarray

def log_coeff(c: complex, n: int):
    result = -0.5*np.abs(c)**2 + n*np.log(c) - 0.5*np.sum(np.log(np.arange(1, n+1)))
    return result

def largest_log_coeff(c: complex):
    n=np.floor(np.abs(c)**2)
    return log_coeff(c,n)

x, xa, xb, t = symbols("x x_a x_b t", real = True)
alpha = symbols("alpha", complex = True)
n = symbols("n", integer = True)
m, omega = symbols("m omega", positive=True)

def coher_coeff_symbol(c, n):
    return coherent_state(n,c)

def coher_coeff_numerical(c, n):
    result = np.exp(-0.5 * np.abs(c) ** 2) * c ** n / np.sqrt(M.factorial(n))
    return result

def coherent_state_my(alpha):
    a = np.abs(alpha)
    length = np.int(np.rint(10*a))
    eigenstates=[]
    coherent_coeff=[]
    m,x,omega = symbols("m x omega")
    c = x - x
    for i in range(length):
        state=psi_n(i, x, m, omega)
        coeff=E**(-0.5*a**2)*alpha**i/sqrt(factorial(i))
        eigenstates.append(state)
        coherent_coeff.append(coeff)
        c+=state*coeff
    return eigenstates, coherent_coeff, c

def coherent_state_time_evolving(alpha):
    a = np.abs(alpha)
    length = int(np.rint(10*a))
    eigenstates=[]
    coherent_coeff=[]
    m,x,t,omega = symbols("m x t omega")
    c = 0
    for i in range(length):
        coeff = M.exp(-0.5*a**2) * alpha**i / M.sqrt(M.factorial(i))
        state=psi_n(i, x, m, omega)*E**(-I * (i+0.5)*omega*t)
        #coeff=E**(-0.5*a**2)*alpha**i/sqrt(factorial(i))
        eigenstates.append(state)
        coherent_coeff.append(coeff)
        c+=coeff*state
    return eigenstates, coherent_coeff, c
