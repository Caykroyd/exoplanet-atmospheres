import core.constants as cst
import numpy as np

def blackbody(T, nu):
    k, h, c = cst.Boltzmann, cst.Plank, cst.SpeedOfLight
    return 2*h*nu**3/c**2 * 1/(np.exp(h*nu/(k*T))-1)
