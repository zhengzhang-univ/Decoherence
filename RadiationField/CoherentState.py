from sympy import *
import numpy as np
import math as M
import scipy.special
from sympy.physics.qho_1d import psi_n
from sympy.physics.quantum.constants import hbar
import matplotlib.pyplot as plt
from matplotlib import cm
init_printing()


x, xa, xb, t = symbols("x x_a x_b t", real = True)
alpha = symbols("alpha", complex = True)
n = symbols("n", integer = True)
m, omega = symbols("m omega", positive=True)

def coher_osci_coeff(c, n):
    """
    c: Complex object. The coherent state parameter.
    n: Integer object. the index/indices of the Fock state(s). Could be a single interger or an interger array.
    """
    result = np.exp(-0.5 * np.abs(c) ** 2) * c ** n / np.sqrt(scipy.special.factorial(n))
    return result

def coherent_state(alpha):
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

def visualize_coherent_state_time_train_2d(alpha):
    eigenfuncs, coeffs, coherent_state = coherent_state_time_evolving(alpha)
    m,t,x,omega = symbols("m t x omega")
    #coherent_state_x_t = N(coherent_state.subs([(m,2*hbar*omega)]))
    coherent_state_x_t = N(coherent_state.subs([(m,2*hbar*omega),(omega,10**6)]))
    f = lambdify([x,t], coherent_state_x_t)
    # time resolution ~ say pi/(5*omega) ~ 10^(-6)
    time_train = np.arange(15)*0.000001
    #x_train = (np.arange(20000)-10000)*0.000000001
    x_train = np.arange(-0.000005,0.000005,0.000000001)
    result = []
    for i in np.arange(len(time_train)):
        func = f(x_train,time_train[i])
        squared_amp = func*np.conjugate(func)
        aux = np.array([ii.real for ii in squared_amp])
        result.append(aux)

    aux=0
    for i in np.arange(len(time_train)):
        plt.plot(x_train,result[i]+ aux,color='black')
        aux+=(np.amax(result[i])+ 0.05*1e7)
    return

def visualize_coherent_state_time_train_3d(alpha):
    eigenfuncs, coeffs, coherent_state = coherent_state_time_evolving(alpha)
    m,t,x,omega = symbols("m t x omega")
    #coherent_state_x_t = N(coherent_state.subs([(m,2*hbar*omega)]))
    coherent_state_x_t = N(coherent_state.subs([(m,2*hbar*omega),(omega,10**6)]))
    f = lambdify([x,t], coherent_state_x_t)
    # time resolution ~ say pi/(5*omega) ~ 10^(-6)
    time_train = np.arange(10)*0.000001
    #x_train = (np.arange(20000)-10000)*0.000000001
    x_train = np.arange(-0.000005,0.000005,0.000000001)
    result = []
    ax = plt.figure().add_subplot(projection='3d')
    for i in np.arange(len(time_train)):
        func = f(x_train,time_train[i])
        squared_amp = func*np.conjugate(func)
        ax.plot(x_train, squared_amp, zs=time_train[i], zdir='x',color='black')
        aux = np.array([ii.real for ii in squared_amp])
        result.append(aux)
    plt.show()
    return


