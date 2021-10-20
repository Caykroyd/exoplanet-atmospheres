from core.transform import Transform
from core.vector import Vector3

import math
import numpy as np
import scipy.integrate

class Camera():

    def __init__(self, parent, focal_length : float = 1, pixel_size : float = 1, view_angle : float = 160, near_clipping_distance : float = 1, frequency_band : tuple = (0,0)):
        self.transform = Transform(parent = parent.transform)
        self._focal_length = focal_length
        self._pixel_size = pixel_size
        self._view_angle = view_angle
        self.eps = near_clipping_distance
        self.frequency_band = frequency_band
        self._vignetting = None
        print(str(self))

    @property
    def pixel_size(self):
        return self._pixel_size

    @pixel_size.setter
    def pixel_size(self, value):
        self._pixel_size = value
        self._vignetting = None

    @property
    def view_angle(self):
        return self._view_angle

    @view_angle.setter
    def view_angle(self, value):
        self._view_angle = value
        self._vignetting = None

    @property
    def focal_length(self):
        return self._focal_length

    @focal_length.setter
    def focal_length(self, value):
        self._focal_length = value
        self._vignetting = None

    def update(self, time, dt):
        # print(self.transform.parent.position, self.transform.position)
        pass

    @property
    def canvas_width(self):
        ideal_width = 2*math.tan(np.deg2rad(self.view_angle/2)) * self.focal_length # in metres
        n_pixels = int(ideal_width / self.pixel_size)
        return n_pixels

    def cull_back(self, X, Y, Z):
        indices = (Z > self.eps)
        return X[indices], Y[indices], Z[indices]

    def project(self, X, Y, Z):
        projection_scale = self.focal_length / self.pixel_size
        return X/Z * projection_scale, Y/Z * projection_scale

    def clip(self, X, Y):
        indices = (np.abs(X) < self.canvas_width/2) & (np.abs(Y) < self.canvas_width/2)
        return X[indices], Y[indices]

    def to_canvas(self, points):
        X, Y, Z = points
        X, Y, Z = self.cull_back(X, Y, Z)
        X, Y = self.project(X, Y, Z)
        X, Y = self.clip(X, Y)
        return X, Y

    def band_integrate(self, I_v):
        I, *_ = scipy.integrate.quad(I_v, *self.frequency_band)
        return I

    def pixel_grid(self):

        ds = self.pixel_size
        N = self.canvas_width

        # (Half) the length of the canvas, in metres
        L = N/2*ds

        # Get center positions of pixels in metres
        u = v = np.linspace(-L + ds/2, L - ds/2, N)

        return np.meshgrid(u,v)

    def vignetting(self, x_c, y_c):


        return ds**2 / np.sqrt(x_c**2 + y_c**2 + self.focal_length**2) / self.focal_length

    def capture(self, I):

        ds = self.pixel_size
        f_0 = self.focal_length

        x, y = self.pixel_grid()
        z = f_0 * np.ones_like(x)

        # rewrite the positions of pixels in terms of the spherical angles
        r = np.stack([x, y, z], axis = -1)
        X, Y, Z = (self.transform.local_to_global_coords(r.reshape(-1, 3)) - self.transform.position).T

        X = X.reshape(x.shape)
        Y = Y.reshape(y.shape)
        Z = Z.reshape(z.shape)

        # r = np.sqrt(Y**2 + Z**2)
        # theta = np.arctan(r)
        # phi = np.arctan2(Y, X)

        I_px = I(X, Y, Z)


        F = I_px * (ds/f_0)**2 / np.linalg.norm(r / f_0, axis=-1)

        return F

    def __str__(self):
        return '''{}
    [Specs]:
        Camera Resolution: {:.2f} Mpx
        Focal Length: {:.1f} mm
        Pixel Size: {:.0f} um
        View Angle: {:.0f} deg
        Near Clipping Distance: {:.2E} m
        '''.format(super().__str__(),self.canvas_width**2 / 1e6, self.focal_length*1e3, self.pixel_size*1e6, self.view_angle, self.eps)
