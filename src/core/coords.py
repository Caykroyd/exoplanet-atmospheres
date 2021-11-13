import numpy as np
from core.vector import Vector3

def to_spherical(x, y, z):
    '''
    Transforms a point from cartesian to spherical coordinates
    '''
    r = np.linalg.norm([x, y, z], axis=0)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return r, theta, phi

def from_spherical(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

def spherical_basis(r, theta, phi):
    '''
    Calculates the basis (e_r, e_theta, e_phi) at given point (r, theta, phi)
    '''
    e_r = Vector3(np.sin(theta) * np.cos(phi),
                  np.sin(theta) * np.sin(phi),
                  np.cos(theta))

    e_theta = Vector3(np.cos(theta) * np.cos(phi),
                      np.cos(theta) * np.sin(phi),
                     -np.sin(theta))

    e_phi = Vector3(- np.sin(phi), np.cos(phi), 0)

    return e_r, e_theta, e_phi
