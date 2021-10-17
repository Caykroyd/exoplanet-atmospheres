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

    @property
    def vignetting(self):
        if self._vignetting is None:
            x, y = self.pixel_grid()
            self._vignetting = self.calculate_vignetting(x, y)

        return self._vignetting

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

    def calculate_vignetting(self, x_c, y_c):

        ds = self.pixel_size
        # x0, x1 = x_c-ds/2, x_c+ds/2
        # y0, y1 = y_c-ds/2, y_c+ds/2

        integrand = lambda x, y: \
            1/np.sqrt(1+x**2)*(np.arctan(x*y / np.sqrt(y**2+1)) - np.arctan(y * np.sqrt(x**2+1))) \
            + np.arcsinh(y) + y * np.log(x + np.sqrt(1 + x**2 + y**2)) - y

        # return integrand(x1,y1) + integrand(x0,y0) - integrand(x0,y1) - integrand(x1,y0)
        return ds**2 / np.sqrt(x_c**2 + y_c**2 + self.focal_length**2) / self.focal_length

    def capture(self, I):

        x, y = self.pixel_grid()
        z = self.focal_length * np.ones_like(x)

        # rewrite the positions of pixels in terms of the spherical angles
        r = np.stack([x, y, z], axis = -1).reshape(-1, 3)
        X, Y, Z = (self.transform.local_to_global_coords(r) - self.transform.position).T

        X = X.reshape(x.shape)
        Y = Y.reshape(y.shape)
        Z = Z.reshape(z.shape)

        # r = np.sqrt(Y**2 + Z**2)
        # theta = np.arctan(r)
        # phi = np.arctan2(Y, X)

        I_px = I(X, Y, Z)

        F = I_px * self.vignetting

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
