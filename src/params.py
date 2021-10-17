from core.vector    import Vector3, Rotation
from core.transform import Transform
from core.scene     import Scene
from core.star      import Star
from core.planet    import Planet
from core.observer  import Observer
from core.camera    import Camera

import yaml

class SceneBuilder:

    def __init__(self, scene):
        self.scene = scene

    def get(self, obj : str, attr : str):
        return getattr(self.scene.objs[obj], attr)

    def set(self, obj : str, attr : str, value):
        setattr(self.scene.objs[obj], attr, value)

    def get_property(self, obj : str, attr : str):
        getter = lambda : self.get(obj, attr)
        setter = lambda val : self.set(obj, attr, val)
        return getter, setter

    def get_float_property(self, obj : str, attr : str, scale : float = 1):
        getter = lambda : self.get(obj, attr) * scale
        setter = lambda val : self.set(obj, attr, val / scale)
        return getter, setter

    @staticmethod
    def from_yaml(file):

        scene = Scene()

        with open(file, 'r') as f:
            params = yaml.load(f)

            # Setup the exoplanet in the scene
            planet = Planet(**params['planet']['params'])
            scene.register(planet, 'planet')

            # Setup the observer on the planet's surface, facing radially
            observer = Observer(planet, **params['observer'])
            scene.register(observer, 'observer')

            cam = Camera(observer, **params['camera'])
            scene.register(cam, 'cam')

            # Setup the star
            star = Star(**params['star']['params'])
            scene.register(star, 'star')

        return SceneBuilder(scene)
