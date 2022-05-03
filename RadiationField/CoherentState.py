import numpy as np
import scipy.special


def coher_osci_amp(c, n):
    """
    c: Complex object. The coherent state parameter.
    n: Integer object. the index/indices of the Fock state(s). Could be a single interger or an interger array.
    """
    result = np.exp(-0.5 * np.abs(c) ** 2) * c ** n / np.sqrt(scipy.special.factorial(n))
    return result
