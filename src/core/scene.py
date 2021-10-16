from collections import defaultdict
import numpy as np

class Scene:
    def __init__(self):
        self.objs = {}

    def register(self, obj, name : str):
        assert name not in self.objs, 'Name already registered in scene!'
        assert obj not in self.objs.values(), 'Object already registered in scene!'
        self.objs[name] = obj

    def destroy(self, name : str):
        assert name in self.objs
        self.objs.pop(name)

    def set_time(self, time : float = 0):
        '''
        Updates all the objects in the scene to time t
        '''
        for obj in self.objs.values():
            obj.update(time, 0)

    def update(self, duration : float = 0, dt : float = 1, ostream = lambda scene,time,dt:{}):
        '''
        Updates all the objects in the scene and saves the output to a stream
        '''
        assert duration >= 0, dt > 0

        output = defaultdict(list)
        for k,v in ostream(self, 0, dt).items():
            output[k].append(v)

        for time in np.arange(0, duration, dt):
            self.set_time(time+dt)
            for k,v in ostream(self, time+dt, dt).items():
                output[k].append(v)
        return output
