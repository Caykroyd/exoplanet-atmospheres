from core.vector import *

class Transform:
    '''
    The base class defining the position and rotation of a body in space.
    This is equivalent to some reference frame at rest (wrt to lab).
    This is done in a hierarchical manner.
    '''
    def __init__(self, local_position : Vector3 = Vector3.zero(), local_rotation : Rotation = Rotation.identity(), parent = None):
        self.local_position = local_position
        self.local_rotation = local_rotation
        self.parent = parent

    def local_to_global_coords(self, point):
        return self.rotation.apply(point) + self.position

    def global_to_local_coords(self, point):
        return self.rotation.inv()(point - self.position)

    @property
    def position(self):
        if self.parent is None:
            return self.local_position
        return self.parent.local_to_global_coords(self.local_position)

    @property
    def rotation(self):
        if self.parent is None:
            return self.local_rotation
        return self.parent.rotation * self.local_rotation

    def translate(self, offset):
        '''
        Applies a translation with an offset in worldcoords
        '''
        self.local_position = parent.global_to_local_coords(self.position + offset)

    def detach(self):
        self.local_position = self.position
        self.local_rotation = self.rotation
        self.parent = None

    def attach(self, parent):
        self.local_position = parent.global_to_local_coords(self.position)
        self.local_rotation = parent.rotation.inverse * self.rotation

    @staticmethod
    def at_origin():
        return Transform(Vector3.zero(), Rotation.identity(), None)
