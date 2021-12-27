import numpy as np
import scipy.spatial.transform

class Vector3(np.ndarray):
    '''
    Override of nd.array class for 3D vectors.
    '''
    def __new__(cls, x, y, z):
        obj = np.stack([x, y, z], axis=0).view(cls)
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return

    @property
    def x(self):
        return self[0]
    @property
    def y(self):
        return self[1]
    @property
    def z(self):
        return self[2]

    def norm(self):
        norm = np.linalg.norm(self, axis=0)
        return norm

    def normalized(self):
        norm = self.norm()
        assert not np.any(np.isclose(norm, 0))
        return self / norm

    @staticmethod
    def project(u, dir):
        dir = dir.normalized()
        return np.dot(u, dir) * dir

    @staticmethod
    def angle(self, u, v):
        return np.arccos(np.dot(u.normalized(), v.normalized()))

    @staticmethod
    def zero():
        return Vector3(0,0,0)

    @staticmethod
    def unit(x, y, z):
        return Vector3(x, y, z).normalized()

class Rotation(scipy.spatial.transform.Rotation):
    '''
    Override of scipy class for representing rotations.
    '''
    @staticmethod
    def from_canonical_to_basis(e_x, e_y, e_z):
        e_x = e_x.normalized()
        e_y = e_y.normalized()
        e_z = e_z.normalized()
        assert np.allclose([e_x @ e_y, e_y @ e_z, e_z @ e_x], 0), 'Basis is not orthogonal!'
        M = np.stack([e_x, e_y, e_z], axis=-1)
        return super(Rotation,Rotation).from_matrix(M)

    @staticmethod
    def from_basis_to_canonical(e_x, e_y, e_z):
        return from_basis_to_canonical().inverse()

    def apply(self, vector : Vector3):
        u = super().apply(vector)
        shape = u.shape
        if(len(shape) < 2):
            return Vector3(*u)
        return u


    def __call__(self, vector : Vector3):
        return self.apply(vector)

def ray_intersects_sphere(origin, dir, center, radius):
    dir = dir.normalized()
    s = center - origin
    s_perp = s - np.dot(s, dir) * dir
    return np.linalg.norm(s_perp) <= radius
