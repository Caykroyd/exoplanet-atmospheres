from core.vector import *

class Transform:
    '''
    The base class defining the position and rotation of a body in space.
    This is equivalent to some reference frame at rest (wrt to lab).
    This is done in a hierarchical manner.
    '''
    def __init__(self, local_position : Vector3 = None, local_rotation : Rotation = None, parent = None):
        if local_position is None:
            local_position = Vector3.zero()
        if local_rotation is None:
            local_rotation = Rotation.identity()
        self.local_position = local_position
        self.local_rotation = local_rotation
        self.parent = parent

    def local_to_global_coords(self, point):
        return self.rotation.apply(point) + self.position

    def global_to_local_coords(self, point):
        return self.rotation.inv().apply(point - self.position)

    def local_to_global_vector(self, vector):
        return self.rotation.apply(vector)

    def global_to_local_vector(self, vector):
        return self.rotation.inv().apply(vector)

    @property
    def position(self):
        if self.parent is None:
            return self.local_position
        return Vector3(self.parent.local_to_global_coords(self.local_position))

    @property
    def rotation(self):
        if self.parent is None:
            return self.local_rotation
        return self.parent.rotation @ self.local_rotation

    def translate(self, offset):
        '''
        Applies a translation with an offset in worldcoords
        '''
        if self.parent is None:
            self.local_position += offset
        else:
            self.local_position = Vector3(self.parent.global_to_local_coords(self.position + offset))

    def detach(self):
        self.local_position = self.position
        self.local_rotation = self.rotation
        self.parent = None

    def attach(self, parent):
        world_position = self.position
        world_rotation = self.rotation
        self.parent = parent
        self.local_position = parent.global_to_local_coords(world_position)
        self.local_rotation = parent.rotation.inv() @ world_rotation

    @staticmethod
    def at_origin():
        return Transform(Vector3.zero(), Rotation.identity(), None)
