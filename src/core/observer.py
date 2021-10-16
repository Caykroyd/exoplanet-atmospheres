from core.planet import Planet

class Observer:

    def __init__(self, planet : Planet, latitude : float = 0, longitude : float = 0):
        self._planet = planet
        self._latitude = latitude
        self._longitude = longitude
        self.transform = self._planet.surface_point(self._latitude, self._longitude)

    def update(self, time, dt):
        pass

    @property
    def latitude(self):
        return self._latitude

    @property
    def longitude(self):
        return self._longitude

    @latitude.setter
    def latitude(self, value):
        self._latitude = value
        transform = self._planet.surface_point(self._latitude, self._longitude)
        self.transform.local_position = transform.local_position
        self.transform.local_rotation = transform.local_rotation

    @longitude.setter
    def longitude(self, value):
        self._longitude = value
        transform = self._planet.surface_point(self._latitude, self._longitude)
        self.transform.local_position = transform.local_position
        self.transform.local_rotation = transform.local_rotation
