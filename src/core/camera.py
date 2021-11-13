from core.transform import Transform
from core.vector    import Vector3
from core.func      import integrate_axis
import core.constants as cst

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

    @property
    def view_angle(self):
        return self._view_angle

    @view_angle.setter
    def view_angle(self, value):
        self._view_angle = value

    @property
    def focal_length(self):
        return self._focal_length

    @focal_length.setter
    def focal_length(self, value):
        self._focal_length = value

    def update(self, time, dt):
        pass

    @property
    def canvas_width(self):
        ideal_width = 2*math.tan(np.deg2rad(self.view_angle/2)) * self.focal_length # in metres
        n_pixels = int(ideal_width / self.pixel_size)
        return n_pixels

    def cull_back(self, X, Y, Z):
        indices = (Z > self.eps)
        return X[indices], Y[indices], Z[indices], indices

    def project(self, X, Y, Z):
        projection_scale = self.focal_length / self.pixel_size
        return X/Z * projection_scale, Y/Z * projection_scale

    def clip(self, X, Y):
        indices = (np.abs(X) < self.canvas_width/2) & (np.abs(Y) < self.canvas_width/2)
        return X[indices], Y[indices], indices

    def to_canvas(self, points):
        X, Y, Z = points
        X, Y, Z, i_ncull = self.cull_back(X, Y, Z)
        X, Y = self.project(X, Y, Z)
        X, Y, i_nclip = self.clip(X, Y)
        
        indices = np.full(i_ncull.shape, False)
        indices[i_ncull][i_nclip] = True
        return X, Y, indices

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

        # Normalize then saturate array
        F = F * 1e0 # W/m2
        F = np.clip(F, 0, 1)
        return F

    def RGB(self, I_v):
        '''
        Computes the RGB values of multiple pixels
        Inputs:
            freq [M]
            I_v  [N, M]

        Output:
            R, G, B [3, N]
        '''

        def compute_basis_XYZ(wvl):

            def piecewise_gaussian(x, mean, std1, std2):
                g1 = np.exp(-0.5*(x - mean)**2/std1**2)
                g2 = np.exp(-0.5*(x - mean)**2/std2**2)
                return np.where(x < mean, g1, g2)

            g = piecewise_gaussian
            wvl = wvl*1e9 # nm

            e_x = 1.056*g(wvl, 599.8, 37.9, 31.0) + 0.362*g(wvl, 442.0, 16.0, 26.7) - 0.065*g(wvl, 501.1, 20.4, 26.2)
            e_y = 0.821*g(wvl, 568.8, 46.9, 40.5) + 0.286*g(wvl, 530.9, 16.3, 31.1)
            e_z = 1.217*g(wvl, 437.0, 11.8, 36.0) + 0.681*g(wvl, 459.0, 26.0, 13.8)

            return e_x, e_y, e_z

        def compute_basis_RGB(wvl):

            e_x, e_y, e_z = compute_basis_XYZ(wvl)

            A = 1/0.17697 * np.array([
                [0.49000, 0.31000, 0.20000],
                [0.17697, 0.81240, 0.01063],
                [0.00000, 0.01000, 0.99000]
            ])

            e_r, e_g, e_b = np.dot(np.linalg.inv(A), [e_x, e_y, e_z])
            return e_r, e_g, e_b

        N = 1000

        freq = np.linspace(*self.frequency_band, N)
        wvl = cst.SpeedOfLight / freq

        e_r, e_g, e_b = compute_basis_RGB(wvl)

        I = I_v(freq)

        R = integrate_axis(I, freq, meas=e_r, axis=-1)
        G = integrate_axis(I, freq, meas=e_g, axis=-1)
        B = integrate_axis(I, freq, meas=e_b, axis=-1)

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.plot(wvl*1e9, e_r)
        # plt.plot(wvl*1e9, e_g)
        # plt.plot(wvl*1e9, e_b)
        # plt.show()

        return R, G, B

    def __str__(self):
        return '''{}
    [Specs]:
        Camera Resolution: {:.2f} Mpx
        Focal Length: {:.1f} mm
        Pixel Size: {:.0f} um
        View Angle: {:.0f} deg
        Near Clipping Distance: {:.2E} m
        '''.format(super().__str__(),self.canvas_width**2 / 1e6, self.focal_length*1e3, self.pixel_size*1e6, self.view_angle, self.eps)
