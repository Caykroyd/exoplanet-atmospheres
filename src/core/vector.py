import numpy as np
import scipy.spatial.transform

class Vector3(np.ndarray):
    '''
    Override of nd.array class for 3D vectors.
    '''
    def __new__(cls, x, y, z):
        obj = np.asarray([x, y, z]).view(cls)
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
        norm = np.linalg.norm(self)
        return norm

    def normalize(self):
        norm = self.norm()
        assert not np.isclose(norm, 0)
        return self / norm

    @staticmethod
    def project(u, dir):
        dir = dir.normalize()
        return np.dot(u, dir) * dir

    @staticmethod
    def angle(self, u, v):
        return np.arccos(np.dot(u.normalized(), v.normalized()))

    @staticmethod
    def zero():
        return Vector3(0,0,0)


class Rotation(scipy.spatial.transform.Rotation):
    '''
    Override of scipy class for representing rotations.
    '''
    @staticmethod
    def from_canonical_to_basis(e_x, e_y, e_z):
        e_x = e_x.normalize()
        e_y = e_y.normalize()
        e_z = e_z.normalize()
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
    dir = dir.normalize()
    s = center - origin
    s_perp = s - np.dot(s, dir) * dir
    return np.linalg.norm(s_perp) <= radius
