import numpy as np
import numpy.random as random

class HGPhaseFunction:
    '''
    HenyeyGreenstein phase function. Used with MIE Scattering.
    '''
    def __init__(self):
        pass

    def g(self, freq):
        return 1

    def get(self, g, mu):
        return 1/2*(1-g**2)/(1 + g**2 - 2*g*mu)**(3/2)

    def sample(self, freq):
        u = random.uniform(0, 1)

        g = self.g(freq)
        u0 = (1-g)/(2*g)
        mu = (1+g**2)/(2*g) - (1-g**2)**2 / (2*g)**3 / (u+u0)**2 # F^{-1}(u), where F(x) = P(X < x)
        return mu


class RayleighPhaseFunction:
    '''
    Rayleigh scattering phase function.
    '''
    @staticmethod
    def get(mu):
        return 3/4*(1+mu**2)

    @staticmethod
    def sample(freq):
        u = random.uniform(0, 1)

        v = np.cbrt(np.sqrt(16*u**2 - 16*u + 5) + 4*u - 2)
        mu = v - 1/v

        return mu
