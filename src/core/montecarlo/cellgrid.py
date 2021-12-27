from core.vector    import Vector3
import core.coords as coords

from math import floor, ceil, sqrt
import numpy as np
import core.montecarlo.sparse as sparse

import sys

class CellGrid_CartesianShell():
    def __init__(self, r0, r1, dx : Vector3):
        N = 2*np.floor(r1/dx).astype(np.int)
        # recalculate exactly
        dx = (2*r1)/N

        self.r0 = r0 # lower bounds of shell
        self.r1 = r1 # upper bounds of shell
        self.dx = dx
        self.N = N

        # print('Constructing sparse 3D cell grid...')
        # self.__construct_sparse_mesh()
        # print('Done.')
        # self.mesh = np.meshgrid(self.x_c, self.y_c, self.z_c)

        print('Cube size:', self.shape)
        print('Volume size of sparse array: {:.0f} / {:.0f}'.format(self.shell_volume, self.cube_volume))

    @property
    def cube_volume(self):
        return np.multiply.reduce(self.shape)

    @property
    def shell_volume(self):
        nx, ny, nz = self.N//2
        mx, my, mz = np.floor(self.r0/self.dx).astype(np.int)
        return int(4/3*np.pi*4*(nx*ny*nz - mx*my*mz))

    def new_sparse_property(self, shape = None, **kwargs):
        if shape is None:
            shape = self.shape
        dims = len(shape)
        prop = sparse.DOKTensor(shape)
        return prop

    def cell_center(self, i, j, k):
        nx, ny, nz = self.N // 2
        dx, dy, dz = self.dx
        return Vector3((i - nx + 0.5)*dx, (j - ny + 0.5)*dy, (k - nz + 0.5)*dz)

    @property
    def cell_volume(self):
        return self.dx.x * self.dx.y * self.dx.z

    @property
    def shape(self):
        return (self.N.x, self.N.y, self.N.z)

    def cell_in_bounds(self, i, j, k):
        '''
        Returns whether given cell(s) indices is inside the grid domain
        '''
        # First check if inside block of shape self.N
        in_block = np.logical_and.reduce((0 <= i, i < self.N.x, 0 <= j, j < self.N.y, 0 <= k, k < self.N.z))

        # Now we create the hash value and check if it's in the hash set
        # hash = self.__cell_to_hash(i, j, k)
        # in_hash = np.isin(hash, self.__keyhash) # numpy version of "x in set"

        nx, ny, nz = self.N // 2
        dx, dy, dz = self.dx

        i, j, k = abs(i-nx), abs(j-ny), abs(k-nz)
        b0 = (i*dx)**2 + (j*dy)**2 + (k*dz)**2
        b1 = ((i+1)*dx)**2 + ((j+1)*dy)**2 + ((k+1)*dz)**2
        # in_sparse = np.logical_and(self.r0**2 <= b0, b1 <= self.r1**2)
        in_sparse = np.logical_or(self.r0**2 <= b1, b0 <= self.r1**2)

        return np.logical_and(in_block, in_sparse)

    def position_to_cell(self, position):
        '''
        Returns the indices of the cell(s) with given positions
        '''
        cell = position // self.dx + self.N//2
        return cell.astype(np.int32)

    def in_bounds(self, position):
        '''
        Returns whether a given position is inside the grid domain
        '''
        # Since broadcasting only works on the highest axis, we need to swap_axis.
        # For clarity, let's do it this way:
        # i, j, k = self.position_to_cell(position)
        # return self.cell_in_bounds(i, j, k)
        return in_shell(position)

    def in_shell(self, position, eps = 1e-9):
        '''
        Returns whether a given position is inside the spherical shell
        '''
        r = position.norm()
        return np.all(np.logical_and(r > self.r0 * (1-eps), r < self.r1 * (1+eps)))

    def project_ray_to_bounds_square(self, start, dir):
        '''
        Returns the first intersection point of a ray with the grid domain (closest to start).
        Ray is defined as a half-segment beginning in start and along dir.
        '''
        if self.in_bounds(start):
            return start

        dir = dir.normalized()

        # Compute the ray distance to intersection on each bounding plane
        t_0p = (+ self.r0 - start) / dir
        t_0m = (- self.r0 - start) / dir
        t_1p = (+ self.r1 - start) / dir
        t_1m = (- self.r1 - start) / dir

        # Join all the distances and get the shortest
        t = min(t for t in (*t0p, *t1p, *t0m, *t1m) if t >= 0 and self.in_bounds(start + t * dir))

        return start + t * dir

    def project_ray_to_shell(self, start, dir):
        '''
        Returns the first intersection point of a ray with the grid domain (closest to start).
        Ray is defined as a half-segment beginning in start and along dir.

        Warning: We only consider outer radius bounds

        Inputs:
            ray start point: [3]
            ray direction: [3]
        Outputs:
            projected posistion: [3]
        '''
        if self.in_shell(start):
            return start

        dir = dir.normalized()

        def radius_intersects(start, dir, r):
            from core.func import quadratic_solution
            # Obtain the positions along the ray corresponding to each radius
            t = quadratic_solution(1, - 2*np.dot(start, dir), np.dot(start, start) - r**2)
            t = np.concatenate(t) # concatenate both solutions

            return t

        # We make a change of coordinates to avoid floating point errors from a far away source
        # This brings us closer to the origin (smallest possible distance to the origin)
        middle = start - np.dot(start, dir)*dir

        t = radius_intersects(middle, - dir, np.array([self.r0, self.r1]))
        pos = middle - np.outer(t, dir)

        # Make sure the positions are contained in the ray semi-segment
        pos = [p for p in pos if np.dot(dir, p - start) >= 0]

        if len(pos)==0:
            print("Warning: ray does not intersect shell.", file=sys.stderr)
            return None

        # Get the closest position to start
        t, pos = min(((p - start).norm(), p) for p in pos)

        assert self.in_shell(pos), 'Sanity check failed! {} < {} < {}'.format(self.r0, pos.norm(), self.r1)
        return pos

    def compute_ray_cells(self, start, end, include_ray_edges = True):
        '''
        Computes the indices of cells through which a given ray crosses,
        and the positions of the intersections.
        If include_edges is True, includes the start and end positions in the
        return array, as long as they are INSIDE the cell bounds.

        Returns:
            cells [C, 3]
            positions [C, 3]
        '''
        dir = (end - start).normalized()

        # Project positions to bounds
        start = self.project_ray_to_shell(start, dir)
        end   = self.project_ray_to_shell(end, -dir)

        if start is None or end is None:  # This edge case can only occur when both positions are outside the shell and the ray does not intersect the shell
            return Vector3([],[],[]), np.array([[],[],[]])

        # Get the cell indices of the start and end positions
        i0, j0, k0 = self.position_to_cell(start)
        i1, j1, k1 = self.position_to_cell(end)

        # Reorder so that the smallest is always first
        min_max = lambda a, b: (min(a, b), max(a,b))
        (i0, i1) = min_max(i0, i1)
        (j0, j1) = min_max(j0, j1)
        (k0, k1) = min_max(k0, k1)

        dx, dy, dz = self.dx
        nx, ny, nz = self.N//2

        # Calculate intersection points with a family of planes perpendicular to each direction: Px, Py, Pz
        tx = ((np.arange(i0+1, i1+1) - nx)*dx - start.x) / (end - start).x
        ty = ((np.arange(j0+1, j1+1) - ny)*dy - start.y) / (end - start).y
        tz = ((np.arange(k0+1, k1+1) - nz)*dz - start.z) / (end - start).z

        # Join all the intersection points,
        # then sort them by how far along the ray they are
        t = np.concatenate([tx, ty, tz])
        assert np.all(t <= 1) and np.all(t >= 0), f"Sanity check failed! You are insane! Better call the coder to fix this bug! t = {t}"

        order = np.argsort(t)
        t = t[order]

        # We now want to calculate the (i,j,k) indices of the cells.
        # Notice that each time the ray intersect with planes Px, Py, Pz,
        # the corresponding index will increment
        diff = np.block([
                [np.ones((i1-i0, 1)),  np.zeros((i1-i0, 1)), np.zeros((i1-i0, 1))],
                [np.zeros((j1-j0, 1)), np.ones((j1-j0, 1)),  np.zeros((j1-j0, 1))],
                [np.zeros((k1-k0, 1)), np.zeros((k1-k0, 1)), np.ones((k1-k0, 1))]]) # (I+J+K, 3)

        if order.size > 0:
            diff = np.take_along_axis(diff, np.expand_dims(order, axis=1), axis = 0) # sorts diff acording to order

        cells = np.array([i0, j0, k0]) + np.cumsum(diff, axis=0)

        if len(order) > 0:
            # Now we filter cells that are not actually in the sparse array
            # since positions may be inside the spherical shell but not in bounds!
            indices = self.cell_in_bounds(*cells.T)

            cells   = cells[indices.nonzero()[0],:]
            t       = t[indices]

        # Only now we calculate the corresponding positions
        pos = start + np.outer(t, end - start)

        if include_ray_edges:
            # Add start/end cells and positions

            if(self.cell_in_bounds(i0, j0, k0)):
                cells = np.concatenate([np.array([[i0, j0, k0]]), cells], axis=0)
                pos = np.concatenate([start[np.newaxis,:], pos], axis=0)

            if(self.cell_in_bounds(i1, j1, k1)):
                # cells = np.concatenate([cells, np.array([[i1, j1, k1]])], axis=0) # End cells are already included by construction!
                pos = np.concatenate([pos, end[np.newaxis,:]], axis=0)
            else:
                cells = cells[:-1] # discard end cell!

        # TODO: Instead of doing this at the end, code properly...
        # cells = np.moveaxis(cells, -1, 0)
        # pos   = np.moveaxis(pos, -1, 0)

        return cells.astype(np.int32), pos
