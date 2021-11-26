import numpy as np
import scipy.integrate

def integrate_axis(y, x, meas=1, axis=0 ):
    integrate = lambda func : scipy.integrate.simpson(func*meas, x)
    return np.apply_along_axis(integrate, axis=axis, arr=y)

def quadratic_solution(a, b, c):
    x0    = -b/(2*a)
    delta = np.sqrt(b**2 - 4*a*c) / (2*a)
    return x0 - delta, x0 + delta
