from core.transform import Transform
from core.vector import Vector3, Rotation

import numpy as np

class Planet():
    '''
    '''
    def __init__(self, season : float = 0, tilt : float = 0, radius : float = 1, rotation_period : float = 0):

        self.tilt = tilt

        self.season = season

        self.transform = Transform(Vector3.zero(), self.R)
        self.radius = radius
        self.rotation_period = rotation_period

    @property
    def tilt_axis(self):
        return Vector3(np.cos(self.season*np.pi/2), np.sin(self.season*np.pi/2), 0)

    @property
    def season_rot(self):
        print('season', self.season)
        Rz = Rotation.from_rotvec(Vector3(0,0,1) * self.season * np.pi/2)
        Rx = Rotation.from_rotvec(self.tilt_axis * np.deg2rad(self.tilt))
        return Rz * Rx

    @property
    def R(self):
        return Rotation.from_rotvec(self.tilt_axis * np.deg2rad(self.tilt))

    @property
    def angular_velocity(self):
        return 2 * np.pi / self.rotation_period

    def update(self, time, dt):
        if self.angular_velocity != 0:
            angular_position = self.angular_velocity * time
            day_rot = Rotation.from_rotvec(Vector3(0,0,1) * angular_position)
            self.transform.local_rotation =  self.R * day_rot

    def surface_point(self, latitude : float, longitude : float):
        '''
        Calculates the position and orientation of an object placed at given
        latitude and longitude and with orientation (e_phi, e_theta, e_r)
        '''
        theta = np.deg2rad(90 - latitude)
        phi   = np.deg2rad(longitude)
        
        e_r, e_theta, e_phi = spherical_basis(self.radius, theta, phi)

        r = Vector3(from_spherical(self.radius, theta, phi))
        q = Rotation.from_canonical_to_basis(e_phi, -e_theta, e_r) # right-handed basis

        return Transform(r, q, parent = self.transform)
