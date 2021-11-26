from core.distributions import blackbody
from core.transform import Transform
from core.vector import Vector3

import core.constants as cst

import numpy as np

class Star:
    def __init__(self, distance : float, temperature : float, radius : float):
        self.transform = Transform()
        self.distance = distance
        self.radius = radius
        self.temperature = temperature

    @property
    def luminosity(self):
        return 4*np.pi*self.radius**2 * cst.Boltzmann * self.temperature**4

    @property
    def distance(self):
        return np.linalg.norm(self.transform.position)

    @distance.setter
    def distance(self, value):
        self.transform.local_position = value * Vector3(1,0,0)

    def spectrum(self, nu):
        return blackbody(self.temperature, nu)

    def sample_frequency(self, freqs):

        f = self.spectrum(freqs)
        F = np.cumsum(f)
        F = F / F[-1]

        u = np.random.uniform(0, 1)
        return np.interp(u, F, freqs) # F^{-1}(u)

    def update(self, time, dt):
        pass
