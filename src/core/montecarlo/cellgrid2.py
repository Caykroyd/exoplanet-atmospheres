import numpy as np
from abc import ABC, abstractmethod

import core.coords as coords
from core.func import quadratic_solution
from math import floor, ceil

class CellGrid(ABC):
    @abstractmethod
    def compute_ray_cells(self, start, end, include_edges = True):
        '''
        Computes the indices of cells through which a given ray crosses,
        and the positions of the intersections.
        If include_edges is True, includes the start and end positions in the
        return array.
        '''
        pass

    @abstractmethod
    def in_bounds(self, position):
        '''
        Returns whether a given position is inside the grid domain
        '''
        pass

    @abstractmethod
    def project_ray_to_bounds(self, start, dir):
        '''
        Returns the first intersection point of a ray with the grid domain (closest to start).
        Ray is defined as a half-segment beginning in start and along dir.
        '''
        pass

    @property
    @abstractmethod
    def shape(self):
        pass

class CartesianCellGrid(CellGrid):
    def __init__(self, r0, r1, dr):
        N  = (r1 - r0)//dr
        dr = (r1 - r0)/ N # recalculate exactly

        self.r0 = r0 # lower bounds
        self.r1 = r1 # upper bounds
        self.dr = dr
        self.N = N
        # Positions of cell centers
        self.x_c = np.linspace(r0.x + dr.x/2, r1.x - dr.x/2, N.x)
        self.y_c = np.linspace(r0.y + dr.y/2, r1.y - dr.y/2, N.y)
        self.z_c = np.linspace(r0.z + dr.z/2, r1.z - dr.z/2, N.z)
        self.mesh = np.meshgrid(self.x_c, self.y_c, self.z_c)

    @property
    def cell_volume(self):
        return self.dr.x * self.dr.y * self.dr.z

    @property
    def shape(self):
        return (*self.N,)

    def in_bounds(self, position):
        '''
        Returns whether a given position is inside the grid domain
        '''
        # Since broadcasting only works on the highest axis, we need to swap_axis.
        # For clarity, let's do it this way:
        x, y, z = position
        r = np.stack([x, y, z], axis=-1)
        return np.all(np.logical_and(r >= self.r0, r <= self.r1), axis=-1)

    def cell_in_bounds(self, index):
        '''
        Returns whether a given cell index is inside the grid domain
        '''
        return np.all(np.logical_and(index >= 0, index < self.N))


    def project_ray_to_bounds(self, start, dir):
        '''
        Returns the first intersection point of a ray with the grid domain (closest to start).
        Ray is defined as a half-segment beginning in start and along dir.
        '''
        if self.in_bounds(start):
            return start

        dir = dir.normalized()

        # Compute the ray distance to intersection on each bounding plane
        t0 = (r0 - start) / dir
        t1 = (r1 - start) / dir

        # Join all the distances and get the shortest
        t = min(t for t in (*t0, *t1) if t >= 0)

        return start + t * dir

    def compute_ray_cells(self, start, end, include_edges = True):
        '''
        Computes the indices of cells through which a given ray crosses,
        and the positions of the intersections.
        If include_edges is True, includes the start and end positions in the
        return array.
        '''
        # A generalization of floor considering a direction
        # We could like to take the integer such that we are still inside the ray
        # Let us consider dx as the direction that points inside the ray
        previous_integer = lambda x, dir : np.where(np.sign(-dir) < 0, np.floor(x), np.ceil(x))

        start_cell = previous_integer((start - self.r0) / self.dr, start - end) + np.sign(start - end)
        end_cell   = previous_integer((end   - self.r0) / self.dr, end - start)

        i0, j0, k0 = start_cell
        i1, j1, k1 = end_cell
        dx, dy, dz = self.dr

        # Calculate intersection points with a family of planes perpendicular to each direction: Px, Py, Pz
        tx = (np.arange(i0, i1, np.sign(end-start))*dx - start.x) / (end - start).x
        ty = (np.arange(j0, j1, np.sign(end-start))*dy - start.y) / (end - start).y
        tz = (np.arange(k0, k1, np.sign(end-start))*dz - start.z) / (end - start).z

        # Join all the intersection points,
        # then sort them by how far along the ray they are
        t = np.concatenate([tx, ty, tz])

        order = np.argsort(t)
        t = t[order]

        # We now want to calculate the (i,j,k) indices of the cells.
        # Notice that each time the ray intersect with planes Px, Py, Pz,
        # the corresponding index will increment
        diff = np.block([
                [np.ones((1, i1-i0)), np.zeros((1, j1-j0)), np.zeros((1, k1-k0))],
                [np.zeros((1, i1-i0)), np.ones((1, j1-j0)), np.zeros((1, k1-k0))],
                [np.zeros((1, i1-i0)), np.zeros((1, j1-j0)), np.ones((1, k1-k0))]])
        diff = np.take_along_axis(diff, order, axis = -1) # sorts diff acording to order

        cells = start_cell + np.cumsum(diff, axis=-1)

        # Add start/end cells and positions
        cells = np.concatenate([start_cell.T, cells, end_cell.T])

        pos = start + t * (end - start)
        pos = np.concatenate([[start], pos, [end]])

        return cells, pos


class SphericalCellGrid(CellGrid):
    def __init__(self, q0, q1, dq = None, N = None):
        '''
            q := (r, theta, phi)
        '''
        assert dq is not None or N is not None, "Must supply either a number of grid divisions or div size!"
        if N is None:
            N  = (q1 - q0)//dq
        dq = (q1 - q0)/ N # recalculate exactly

        self.q0 = q0 # lower bounds
        self.q1 = q1 # upper bounds
        self.dq = dq
        self.N = N
        # Positions of cell centers
        self.r_c     = np.linspace(q0.x + dq.x/2, q1.x - dq.x/2, N.x)
        self.theta_c = np.linspace(q0.y + dq.y/2, q1.y - dq.y/2, N.y)
        self.phi_c   = np.linspace(q0.z + dq.z/2, q1.z - dq.z/2, N.z)
        self.mesh = np.meshgrid(self.r_c, self.theta_c, self.phi_c)

    @property
    def shape(self):
        return (*self.N,)

    def cell_volume(self, i, j, k):
        r_c     = self.r_c[i]
        theta_c = self.theta_c[j]
        phi_c   = self.phi_c[k]

        dr, dtheta, dphi = self.dq

        return (r_c**2 + 1/3*dr**2) * 2*np.sin(theta_c) * dr * np.sin(dtheta) * dphi

    def in_bounds(self, position, threshold = 1e-6):
        '''
        Returns whether a given position is inside the grid domain
        '''
        r, theta, phi = coords.to_spherical(*position)
        q = np.stack([r, theta, phi], axis=-1)

        eps = threshold
        return np.all(np.logical_and(q >= self.q0 * (1-eps), q <= self.q1 * (1+eps)), axis=-1)

    def cell_in_bounds(self, index):
        '''
        Returns whether a given cell index is inside the grid domain
        '''
        return np.all(np.logical_and(index >= 0, index < self.N))

    def project_ray_to_bounds(self, start, dir):
        '''
        Returns the first intersection point of a ray with the grid domain (closest to start).
        Ray is defined as a half-segment beginning in start and along dir.

        ! Warning: At the moment, we only consider outer radius bounds !
        '''
        if self.in_bounds(start):
            return start

        dir = dir.normalized()

        def radius_intersects(start, dir, r):
            # Obtain the positions along the ray corresponding to each radius
            t = quadratic_solution(1, - 2*np.dot(start, dir), np.dot(start, start) - r**2)
            t = np.concatenate(t) # concatenate both solutions

            return t

        r0, _, _ = self.q0
        r1, _, _ = self.q1

        # We make a change of coordinates to avoid floating point errors from a far away source
        # This brings us closer to the origin (smallest possible distance to the origin)
        middle = start - np.dot(start, dir)*dir

        t_r = radius_intersects(middle, - dir, np.array([r0, r1]))

        pos_r = middle - np.outer(t_r, dir)

        # get the closest position to start, contained in the ray semi-segment
        _, pos = min((np.linalg.norm(p - start), p) for p in pos_r if np.dot(dir, p - start) >= 0)

        assert self.in_bounds(pos), 'Sanity check failed! {} < {} < {}'.format(r0, pos.norm(), r1)
        return pos

    def compute_ray_cells(self, start, end, include_edges = True):
        '''
        Computes the indices of cells through which a given ray crosses,
        and the positions of the intersections.
        If include_edges is True, includes the start and end positions in the
        return array.

        Warning: This function will give you a heart attack. It's already too late for me.
        '''
        # Change to spherical coordinates
        r_s, theta_s, phi_s = coords.to_spherical(*start)
        r_e, theta_e, phi_e = coords.to_spherical(*end)

        dir = (end - start).normalized()

        r0, theta0, phi0 = self.q0
        r1, theta1, phi1 = self.q1
        dr, dtheta, dphi = self.dq
        Nr, Ntheta, Nphi = self.N

        def radius_intersects(start, end):
            '''
            Returns the position along the ray where it intersects the family of radial spheres
            '''
            dir = (end - start).normalized()
            ray_length = (end - start).norm()
            # Let's first find the range of 'radius' values crossed by this FINITE ray
            r_max = max(r_s, r_e)
            # Find r_min (a bit more challenging)
            t_r_min = - np.dot(start, dir)

            if t_r_min > 0 and t_r_min < ray_length:
                r_min = (start + t_r_min * dir).norm()
            else:
                r_min = min(r_s, r_e)

            # Now take the grid indices of these values
            i0 = floor((r_min - r0)/dr) + 1
            i1 = ceil((r_max - r0)/dr) + 1
            i0, i1 = np.clip([i0, i1], 1, Nr)

            # Calculate the range of radius values: all of these radii intersect the ray
            R = r0 + np.arange(i0, i1)*dr
            print(f'Obtained radius interval [{r_min:.4E}, {r_max:.4E}], r0 = {r0:.2E}', dr, i0, i1)
            print("dR", R)
            # Obtain the positions along the ray corresponding to each radius
            t = quadratic_solution(1, -2*t_r_min, np.dot(start, start) - R**2)
            print("t", t)
            t = np.concatenate(t) # concatenate both solutions
            t = np.sort(t) # and sort along the ray
            print("t", t)
            assert len(t) == 2*len(R), "Sanity check failed"

            # we know there will always be (i1-i0) decresing radii followed by (i1-i0) increasing radii
            diff = np.concatenate([-np.ones(i1 - i0), np.ones(i1 - i0)])

            # remove solutions outside the ray or bounds
            filter = np.where(np.logical_and.reduce([t > 0, t < ray_length, self.in_bounds((start + np.outer(t, dir)).T)]))
            t    = t[filter]
            diff = diff[filter]

            return t, diff

        def polar_intersects(start, end):
            '''
            Returns the position along the ray where it intersects the family of polar-angle cones
            '''
            dir = (end - start).normalized()
            ray_length = (end - start).norm()
            # Let's first find the range of 'theta' values crossed by this FINITE ray
            theta_max = max(theta_s, theta_e)
            # Find r_min (a bit more challenging)
            S0, S = start, end-start
            t_th_min = (np.dot(S, S0)*S0.z - np.dot(S0,S0)*S.z)/(2*np.dot(S, S0)*S.z - np.dot(S,S)*S0.z)
            if t_th_min > 0 and t_th_min < ray_length:
                pos_th_min = t_th_min * (end - start) + start
                _, theta_min, _ = coords.to_spherical(*pos_th_min)
            else:
                theta_min = min(theta_s, theta_e)

            # Now take the grid indices of these values
            j0 = floor((theta_min - theta0)/dtheta) + 1
            j1 = ceil((theta_max - theta0)/dtheta) + 1
            j0, j1 = np.clip([j0, j1], 0, Ntheta)

            # Calculate the range of radius values: all of these theta intersect the ray (intermediate value theorem)
            Theta = np.arange(j0, j1)*dtheta

            # Obtain the positions along the ray corresponding to each theta
            Z, Z0 = S*np.cos(Theta), S0*np.cos(Theta)
            t = quadratic_solution(np.dot(Z, Z) - S.z**2, 2*np.dot(Z, Z0)-2*S.z*S0.z, np.dot(Z0, Z0) - S0.z**2)
            t = np.concatenate(t) # concatenate both solutions
            t = np.sort(t) # and sort along the ray

            # we know there will always be (i1-i0) decresing radii followed by (i1-i0) increasing radii
            diff = np.concatenate([-np.ones(i1 - i0), np.ones(i1 - i0)])

            # remove solutions outside the ray or bounds
            filter = np.where(np.logical_and.reduce([t > 0, t < ray_length, self.in_bounds((start + np.outer(t, dir)).T)]))
            t    = t[filter]
            diff = diff[filter]

            return t, diff
            return t, diff

        def azimuth_intersects(start, end):
            '''
            Returns the position along the ray where it intersects the family of azimuthal planes
            '''
            dir = (end - start).normalized()
            ray_length = (end - start).norm()
            # Let's first find the range of 'phi' values crossed by this FINITE ray
            delta_phi = (phi_e - phi_s) % (2*np.pi)
            if delta_phi <= np.pi:
                phi_max = phi_e
                phi_min = phi_max - delta_phi
            else:
                phi_max = phi_s
                phi_min = phi_max - (np.pi - delta_phi)

            # Now take the grid indices of these values
            k0 = floor((phi_min - phi0)/dphi) + 1
            k1 = ceil((phi_max - phi0)/dphi) + 1
            k0, k1 = np.clip([i0, i1], 0, Nphi)

            # Calculate the range of phi values: all of these phi intersect the ray
            Phi = np.arange(k0, k1)*dphi

            # Obtain the positions along the ray corresponding to each Phi
            t = - (start.y - start.x * np.tan(Phi)) / ((end-start).y - (end-start).x * np.tan(Phi))
            t = np.concatenate(t) # concatenate both solutions
            t = np.sort(t) # and sort along the ray

            diff = np.ones(i1 - i0) * (1 if delta_phi <= np.pi else -1)

            # remove solutions outside the ray or bounds
            filter = np.where(np.logical_and.reduce([t > 0, t < ray_length, self.in_bounds((start + np.outer(t, dir)).T)]))
            t    = t[filter]
            diff = diff[filter]

            return t, diff

        t_r, diff_r = radius_intersects(start, end)
        t_th, diff_th = polar_intersects(start, end)
        t_phi, diff_phi = azimuth_intersects(start, end)

        # Join all the intersection points,
        # then sort them by how far along the ray they are
        t = np.concatenate([t_r, t_th, t_phi])

        order = np.argsort(t)
        t = t[order]

        # We now want to calculate the (i,j,k) indices of the cells.
        # Notice that each time the ray intersect with planes Px, Py, Pz,
        # the corresponding index will increment
        di, dj, dk = len(t_r), len(t_th), len(t_phi)
        diff = np.block([
                [diff_r,            np.zeros((1, dj)), np.zeros((1, dk))],
                [np.zeros((1, di)), diff_th,           np.zeros((1, dk))],
                [np.zeros((1, di)), np.zeros((1, dj)), diff_phi]])
        diff = np.take_along_axis(diff, order, axis = -1) # sorts diff acording to order

        # Compute the cells as the discrete integral of diffs
        start_cell = floor((start - self.q0) / self.dq)

        cells = start_cell + np.cumsum(diff, axis=-1)

        # Add start cell (end cell is already included, I think)
        cells = np.concatenate([start_cell.T, cells])

        q_s = np.stack([r_s, theta_s, phi_s], axis=0)
        q_e = np.stack([r_e, theta_e, phi_e], axis=0)

        pos = q_s + t * (q_e - q_s)
        pos = coords.from_spherical(*pos)
        pos = np.concatenate([[start], pos, [end]])

        return cells, pos
