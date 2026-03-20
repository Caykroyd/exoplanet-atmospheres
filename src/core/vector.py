import numpy as np
import scipy.spatial.transform as scipyt

class Vector3(np.ndarray):
    '''
    Override of nd.array class for 3D row vectors.
    '''
    def __new__(cls, *args):
        if len(args) == 1:
            arr = np.asarray(args[0])
            if arr.shape[-1] != 3:
                raise ValueError(f"Expected last dimension to have size 3, got shape {arr.shape}")
            obj = arr.view(cls)

        elif len(args) == 3:
            x, y, z = args
            obj = np.stack([x, y, z], axis=-1).view(cls)

        else:
            raise TypeError("Vector3 expects either (x, y, z) or array with shape (..., 3).")

        return obj

    def __array_finalize__(self, obj):
        if obj is None: return

    @property
    def x(self):
        return self[..., 0]
    @property
    def y(self):
        return self[..., 1]
    @property
    def z(self):
        return self[..., 2]

    def norm(self):
        norm = np.linalg.norm(self, axis=-1)
        return norm

    def normalized(self):
        norm = self.norm()[..., None]
        assert not np.any(np.isclose(norm, 0))
        return self / norm

    @staticmethod
    def project(u : Vector3, dir : Vector3):
        dir = dir.normalized()
        dot = np.einsum('...i,...i->...', u, dir)
        return dot[..., None] * dir

    @staticmethod
    def angle(u, v):
        dot = np.einsum('...i,...i->...', u.normalized(), v.normalized())
        return np.arccos(dot)

    @staticmethod
    def zero():
        return Vector3(0,0,0)

    @classmethod
    def unit(cls, *args):
        return cls(*args).normalized()

class Rotation():
    '''
    Override of scipy class for representing rotations.
    '''
    def __init__(self, rot):
        if not isinstance(rot, scipyt.Rotation):
            raise TypeError(f'Expected scipy Rotation, got {type(rot)!r}')
        self.data = rot

    @classmethod
    def from_quat(cls, quat, *args, **kwargs):
        return cls(scipyt.Rotation.from_quat(quat, *args, **kwargs))

    @classmethod
    def from_rotvec(cls, rotvec, *args, **kwargs):
        return cls(scipyt.Rotation.from_rotvec(rotvec, *args, **kwargs))

    @classmethod
    def from_euler(cls, seq, angles, *args, **kwargs):
        return cls(scipyt.Rotation.from_euler(seq, angles, *args, **kwargs))

    @classmethod
    def from_matrix(cls, matrix, *args, **kwargs):
        return cls(scipyt.Rotation.from_matrix(matrix, *args, **kwargs))

    @classmethod
    def from_basis(cls, e_x, e_y, e_z):
        e_x = e_x.normalized()
        e_y = e_y.normalized()
        e_z = e_z.normalized()
        assert np.allclose([e_x @ e_y, e_y @ e_z, e_z @ e_x], 0), 'Basis is not orthogonal!'
        M = np.stack([e_x, e_y, e_z], axis=-1)
        assert np.linalg.det(M) > 0, "Basis is left-handed, not a proper rotation!"
        return cls.from_matrix(M)

    @classmethod
    def identity(cls, *args, **kwargs):
        return cls(scipyt.Rotation.identity(*args, **kwargs))

    def inv(self):
        return Rotation(self.data.inv())

    def apply(self, vector : Vector3, inverse = False):
        u = self.data.apply(vector, inverse)
        return Vector3(u)

    def __call__(self, vector : Vector3):
        return self.apply(vector)

    def __matmul__(self, other):        
        if isinstance(other, Rotation):
            return self.__class__(self.data * other.data)
        if isinstance(other, Vector3):
            return self.apply(other)
        return NotImplemented
    
def ray_intersects_sphere(origin, dir, center, radius):
    dir = dir.normalized()
    s = center - origin
    s_perp = s - np.dot(s, dir) * dir
    return np.linalg.norm(s_perp) <= radius
