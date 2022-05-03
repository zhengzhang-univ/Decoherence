import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def two_oscillators(y, t, omega1, omega2, m1, m2, a):
    phi, z1, chi, z2 = y
    dydt = [z1, -omega1**2*phi - (m2*omega2**2/(m1*a))*chi*phi, z2, -omega2**2*chi - omega2**2*phi**2/(2*a)]
    return dydt
