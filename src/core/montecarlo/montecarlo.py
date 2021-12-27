from core.vector    import Vector3
from core.transform import Transform

from core.montecarlo.photon     import PhotonPacket
from core.montecarlo.cellgrid   import CellGrid_CartesianShell

from core.montecarlo.phase_function import RayleighPhaseFunction

import numpy as np
import numpy.random as random

import sys

class MonteCarlo:

    def __init__(self, planet, atmosphere, camera, star):
        # The grid should placed at the planet center
        self.transform = Transform(local_position = planet.transform.position)

        self.planet = planet
        self.atmosphere = atmosphere
        self.camera = camera
        self.lightsource = star

        self.__grid_params = None
        self.__grid = None

    def set_params(self, n_packets, grid_blocks, n_freqs):
        self.n_packets = n_packets
        self.grid_blocks = np.array(grid_blocks)
        self.n_freqs = n_freqs

    def update(self, time, dt):
        pass

    @property
    def grid(self):
        # Calculate grid parameters
        height  = self.atmosphere.depth

        r0 = self.planet.radius
        r1 = self.planet.radius + height

        N = self.grid_blocks
        dx = Vector3(r1, r1, r1) / N

        grid_params = {'r0': r0, 'r1': r1, 'dx': dx}

        def compare_dicts(dict1, dict2):
            '''
            Compare dicts in a way that works with numpy arrays
            '''
            if dict1 is None or dict2 is None:
                return False
            if dict1.keys() != dict2.keys():
                return False
            return all(np.all(dict1[key] == dict2[key]) for key in dict1.keys())

        # If these values changed, we need a new grid
        if compare_dicts(grid_params, self.__grid_params) is False:
            self.__grid_params = grid_params
            self.__grid = CellGrid_CartesianShell(**self.__grid_params)

        return self.__grid

    def run(self, n_packets = None):
        '''
        Run the MonteCarlo method for source functions facing the observer
        '''
        if n_packets is None:
            n_packets = self.n_packets

        self.freqs = np.linspace(*self.camera.frequency_band, self.n_freqs)
        self.S = self.grid.new_sparse_property(shape=(*self.freqs.shape, *self.grid.shape)) # source function

        packet = self.create_photon_packets(self.lightsource, n_packets)

        for i, photon in enumerate(packet):
            while(photon.position is not None and self.grid.in_shell(photon.position)):
                self.next_event(photon)
            if (i * 100) % n_packets == 0:
                print(f"{(i * 100) // n_packets + 1}% done...")

        print("MonteCarlo Complete!")
        print(f"Number of cells visited: {self.S.nnz} / {self.grid.shell_volume}")
        print(f"Maximum value of Source function {self.S.tobsr().max()}")

        return self.S

    def create_photon_packets(self, lightsource, N):

        print(f'Running Monte Carlo with {N} photons.')
        packet = [None for _ in range(N)]

        src_pos = lightsource.transform.position
        obs_pos = self.camera.transform.position
        origin  = self.planet.transform.position
        target  = origin

        src_dist = np.linalg.norm(src_pos - origin)
        radius = self.planet.radius + self.atmosphere.depth
        aperture = np.arctan(radius / src_dist)

        # Use smaller aperture: target the camera specifically
        target = obs_pos
        aperture /= 1e3

        for i in range(N):

            frequency = lightsource.sample_frequency(self.freqs)

            position = src_pos + lightsource.radius * sample_uniform_cone(target - src_pos, np.pi/2)

            # random but well-chosen, a cone
            direction = sample_uniform_cone(target - position, aperture)

            # Transform to local coords
            position = position - origin
            # Project to atmosphere
            position = self.grid.project_ray_to_shell(position, direction) # project to atmosphere

            if position is None:
                packet[i] = PhotonPacket(None, None, None, None)
                continue # sample a new packet

            mu = np.dot(direction, (position - src_pos).normalized())
            P_emission = (1/2) * mu * (1-np.cos(aperture))

            luminosity = lightsource.luminosity / N * P_emission

            packet[i] = PhotonPacket(position, direction, frequency, luminosity)

        return packet

    def next_event(self, photon):

        start = photon.position
        dir = photon.direction
        freq = photon.frequency

        optical_depth = self.sample_optical_depth()
        # print("Sampled Optical Depth:", optical_depth)

        a_scat = self.atmosphere.coef_scatter(freq)

        end = self.atmosphere.optical_depth_to_position(start, dir, optical_depth, freq) # meybe it's easier to compute the "vertical optical depth" v_tau = tau * mu, ang then get the output height, and transform back into position

        cells, positions = self.grid.compute_ray_cells(start, end)

        if cells.size == 0: # The photon is outside the domain so we dont care about it so it must be destroyed!
            print("Warning: Photon does not intersect domain. Destroying it...", file=sys.stderr)
            photon.position = None
            return

        ds = np.linalg.norm(np.diff(positions, axis=0), axis=-1)

        # For each cell, use the "average" along the cell - this should be in atmosphere code
        positions = (positions[1:] + positions[:-1])/2

        # obs_pos = self.transform.global_to_local_coords(self.camera.transform.position)
        obs_pos = self.camera.transform.position

        mu = np.einsum("i,ni->n", dir, (obs_pos - positions)/np.linalg.norm(obs_pos - positions, axis=-1)[...,np.newaxis])
        dV = self.grid.cell_volume

        dS = ds * a_scat * photon.luminosity / (4*np.pi*dV) * 2*RayleighPhaseFunction.get(mu)

        freq_bin = np.digitize(photon.frequency, self.freqs)
        freq_bin = np.clip(freq_bin, 0, len(self.freqs))

        i, j, k = np.moveaxis(cells, -1, 0)
        self.S[freq_bin, i, j, k] += dS

        photon.position = end
        photon.direction = self.random_scattered_direction(photon)

    def random_scattered_direction(self, photon):
        '''
        Randomly scatters the photon from incoming direction following an angular probability distribution.
        '''
        phi = random.uniform(0, 2*np.pi)
        mu  = RayleighPhaseFunction.sample(photon.frequency)

        # Compute a uniformly random normal direction
        nx, ny = perpendicular_basis(photon.direction)
        normal = nx*np.cos(phi) + ny*np.sin(phi);

        # final direction is a deviation from the original direction towards the normal
        return photon.direction * mu + normal * np.sqrt(1 - mu**2);

    def sample_optical_depth(self):
        '''
        Randomly selects a distance traveled by the photon before next scatter/absorption event.
        Event is a poisson process with lambda = 1, returned in units of optical depth
        '''
        return random.exponential(1)

    # def is_true_scatter_event(self, photon):
    #     a_abs = self.coef_absorption(photon.freq)
    #     a_scat = self.coef_scatter(photon.freq)
    #     scatter_chance = a_scat / (a_abs + a_scat)
    #     return random.binomial(1, scatter_chance)

    # def random_emission_direction(self):
    #     '''
    #     Uniform phase function
    #     '''
    #     phi = random.uniform(0, 2*np.pi)
    #
    #     u = random.uniform(0, 1)
    #     mu = np.sqrt(u)
    #
    #     x = mu * np.cos(phi)
    #     y = mu * np.sin(phi)
    #     z = np.sqrt(1 - mu**2);
    #
    #     return np.array([x, y, z])

def perpendicular_basis(v, eps=1e-8):
    '''
    Finds two arbitrary unit vectors normal to v.
    Assumes v is normalized and non-zero.
    '''
    ix, iy, _ = np.argsort(np.abs(v)) # indices of smallest elements in the array
    dx, dy = np.eye(1, 3, ix).flatten(), np.eye(1, 3, iy).flatten()
    return Vector3(*np.cross(v, dx)).normalized(), Vector3(*np.cross(v, dy)).normalized()


def sample_uniform_cone(axis, aperture):

    axis = axis.normalized()

    phi = random.uniform(0, 2*np.pi)
    mu  = random.uniform(np.cos(aperture), 1)

    # Compute a uniformly random normal direction
    nx, ny = perpendicular_basis(axis)
    normal = nx*np.cos(phi) + ny*np.sin(phi);

    # final direction is a deviation from the original direction towards the normal
    return axis * mu + normal * np.sqrt(1 - mu**2);
