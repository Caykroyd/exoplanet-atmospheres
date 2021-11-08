import numpy as np
import scipy.integrate

def integrate_axis(y, x, meas=1, axis=0 ):
    integrate = lambda func : scipy.integrate.simpson(func*meas, x)
    return np.apply_along_axis(integrate, axis=axis, arr=y)
