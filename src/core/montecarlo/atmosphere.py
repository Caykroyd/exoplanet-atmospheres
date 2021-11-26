from core.vector    import Vector3
from core.transform import Transform


import numpy as np
import core.constants as cst

class Species:
    def __init__(self, fraction, rayleigh_cross_section):
        self.fraction = fraction
        self._rayleigh_cross_section = rayleigh_cross_section

    def rayleigh_cross_section(self, freq):
        wvl = cst.SpeedOfLight / freq
        return self._rayleigh_cross_section * (532e-9 / wvl)**4


class ConstantBlockAtmosphere:
    '''
    A basic atmosphere consisting of a paralelepiped block with a given width around the origin,
    and height from the origin
    '''
    def __init__(self, parent, species, density, depth):
        self.transform = Transform(parent=parent)
        self.species = species
        self.density = density
        self.depth = depth

    def update(self, time, dt):
        pass

    @property
    def density(self):
        return self._density

    @density.setter
    def density(self, value):
        self._density = lambda pos : value# * np.ones_like(pos[0])

    @property
    def species(self):
        return self._species

    @species.setter
    def species(self, value):
        self._species   = [Species(**params) for name, params in value.items()]

    def coef_scatter(self, position : Vector3, freq):
        '''
        I think species must be treated differently and instead of adding their coefficients
        '''
        A_s = (s.rayleigh_cross_section(freq) for s in self.species)
        X   = (s.fraction for s in self.species)
        N = self.density(position)
        return  N * sum(a_s * x for a_s, x in zip(A_s, X))

    def coef_absorption(self, position : Vector3, freq):
        return 0

    def optical_depth_to_position(self, start_pos, dir, optical_depth, freq):
        '''
        This function supposes that the scatter coefficient is constant.
        TODO: Use integration to determine the exact position
        '''
        return start_pos + dir * (optical_depth / self.coef_scatter(start_pos, freq))
