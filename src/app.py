import sys

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon, QFont, QIntValidator

from qt.tabs import Tabbar, Tab
from qt.canvas import PyplotCanvas
from qt.widget import WidgetGroup, Slider, FloatField, RangeField

from params import SceneBuilder

class App(QtWidgets.QMainWindow):
    def __init__(self, builder : SceneBuilder):
        super().__init__()
        self.title = 'Exoplanet atmosphere simulator'

        self.tabs = Tabbar([
            SimulationTab('Simulation', builder),
            SetupTab('Scene Setup', builder),
            TrajectoryTab('Trajectory', builder),
            SpectrumTab('Light Spectrum', builder)
            ],
        self)

        self.setCentralWidget(self.tabs)

        self.show()

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, value):
        self._title = value
        self.setWindowTitle(value)

InspectorGroup = lambda title, widget : WidgetGroup(QHBoxLayout(), [QLabel(title), widget])

class SetupTab(Tab):

    def __init__(self, name, builder):

        self.plot = PyplotCanvas(self, width=5, height=4, dpi=100, proj='3d')

        self.builder = builder

        title = QLabel('Parameters')
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setFont(QFont("Arial", 12, weight=QFont.Bold))

        subtitle1 = QLabel('Planetary')
        subtitle1.setFont(QFont("Arial", weight=QFont.Bold))
        tilt   = FloatField((-90,90,2), *builder.get_property('planet', 'tilt'), self.refresh)
        season = Slider(QtCore.Qt.Horizontal, (0,4,0.01), *builder.get_property('planet', 'season'), self.refresh)

        subtitle2 = QLabel('Observer')
        subtitle2.setFont(QFont("Arial", weight=QFont.Bold))

        latitude  = FloatField((-90.,90.,2), *builder.get_property('observer', 'latitude'), self.refresh)
        longitude = FloatField((-180,180,2), *builder.get_property('observer', 'longitude'), self.refresh)

        self.sidebar = WidgetGroup(QVBoxLayout(),
            [
            QLabel(' '),
            title,
            QLabel(' '),
            subtitle1,
            InspectorGroup('Tilt Angle (°):', tilt),
            InspectorGroup('Season:', season),
            QLabel(' '),
            subtitle2,
            InspectorGroup('Latitude (°):', latitude),
            InspectorGroup('Longitude (°):', longitude),
            None],
            )


        super().__init__(name, QHBoxLayout(), [self.plot, self.sidebar])

    def init(self):
        # TOOD: pull initial values into fields
        self.refresh()

    def refresh(self):
        self.plot.refresh()

    def set_update_callback(self, callback):
        self.plot.callback = callback
        self.init()

class TrajectoryTab(Tab):

    def __init__(self, name, builder):

        self.builder = builder

        self.plot = PyplotCanvas(self, width=5, height=4, dpi=100)

        super().__init__(name, QHBoxLayout(), [self.plot])

    def init(self):
        # TOOD: pull initial values into fields
        self.refresh()

    def refresh(self):
        self.plot.refresh()

    def set_update_callback(self, callback):
        self.plot.callback = callback
        self.init()

class SimulationTab(Tab):

    def __init__(self, name, builder):

        self.builder = builder
        self.plot = PyplotCanvas(self, width=5, height=4, dpi=100)

        FieldGroup = lambda title, widget : WidgetGroup(QHBoxLayout(), [QLabel(title), widget])

        self.time_slider = QSlider(QtCore.Qt.Horizontal)
        self.time_slider.setMaximum(self.builder.get('planet','rotation_period'))
        self.time_slider.valueChanged.connect(self.set_time)
        self.play_button = QPushButton('Pause')
        self.play_button.clicked.connect(self.on_play_pause)

        self.bottombar = WidgetGroup(QVBoxLayout(),
            [self.time_slider,
            self.play_button,
            None],
            )

        title = QLabel('Parameters')
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setFont(QFont("Arial", 12, weight=QFont.Bold))

        subtitle1 = QLabel('Star')
        subtitle1.setFont(QFont("Arial", weight=QFont.Bold))

        distance = FloatField((0,10,2), *builder.get_float_property('star', 'distance', scale=1/1.496e+11), self.refresh)
        radius   = FloatField((0,1e10,2), *builder.get_float_property('star', 'radius', scale=1/6.957e+8), self.refresh)

        subtitle2 = QLabel('Camera')
        subtitle2.setFont(QFont("Arial", weight=QFont.Bold))

        pixel_size  = FloatField((0.01,1000,2), *builder.get_float_property('cam', 'pixel_size', scale=1e6), self.refresh)
        view_angle = FloatField((0,180,2), *builder.get_property('cam', 'view_angle'), self.refresh)

        self.sidebar = WidgetGroup(QVBoxLayout(),
            [
            QLabel(' '),
            title,
            QLabel(' '),
            subtitle1,
            InspectorGroup('Radius (Solar Radii):', radius),
            InspectorGroup('Distance (AU):', distance),
            QLabel(' '),
            subtitle2,
            InspectorGroup(r'Pixel Size (micron):', pixel_size),
            InspectorGroup('View Angle (°):', view_angle),
            None],
            )
        flag = False
        super().__init__(name, QHBoxLayout(), [WidgetGroup(QVBoxLayout(), [self.plot, self.bottombar]), self.sidebar])

    def set_update_callback(self, func):
        self.plot.callback = func
        self.plot.refresh()

    def refresh(self):
        self.plot.refresh()

    def set_timer(self, tick=240250, dt = 100):
        self.time = 0
        self.timer = QtCore.QTimer(self, interval=tick, timeout=lambda : self.set_time(self.time+dt))
        # self.timer.start()
        self.is_playing = True

    def set_time(self, time):
        self.time = time % self.time_slider.maximum()
        self.time_slider.setValue(self.time)
        self.builder.scene.set_time(self.time)
        self.refresh()

    def pause(self):
        self.timer.stop()
        self.play_button.setText('Play')
        self.is_playing = False

    def play(self):
        self.timer.start()
        self.play_button.setText('Pause')
        self.is_playing = True


    def on_play_pause(self):
        if self.is_playing:
            self.pause()
        else:
            self.play()

class SpectrumTab(Tab):

    def __init__(self, name, builder):

        self.builder = builder

        self.plot = PyplotCanvas(self, width=5, height=4, dpi=100)

        title = QLabel('Parameters')
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setFont(QFont("Arial", 12, weight=QFont.Bold))

        subtitle1 = QLabel('Star')
        subtitle1.setFont(QFont("Arial", weight=QFont.Bold))

        temperature = FloatField((0,1e5,2), *builder.get_property('star', 'temperature'), self.refresh)

        subtitle2 = QLabel('Camera')
        subtitle2.setFont(QFont("Arial", weight=QFont.Bold))

        frequency_band = RangeField((0.1,1e10, 1), *builder.get_property('cam', 'frequency_band'), self.refresh, scale=1e-12)

        self.sidebar = WidgetGroup(QVBoxLayout(),
            [
            QLabel(' '),
            title,
            QLabel(' '),
            subtitle1,
            InspectorGroup('Temperature (K):', temperature),
            QLabel(' '),
            subtitle2,
            InspectorGroup('Band (THz):', frequency_band),
            None],
            )

        super().__init__(name, QHBoxLayout(), [self.plot, self.sidebar])

    def init(self):
        # TOOD: pull initial values into fields
        self.refresh()

    def refresh(self):
        self.plot.refresh()

    def set_update_callback(self, callback):
        self.plot.callback = callback
        self.init()

from params import SceneBuilder
from plot import *

import yaml
import sys

if __name__=='__main__':

    file = './in/config.yaml'

    builder = SceneBuilder.from_yaml(file)
    scene = builder.scene

    app = QApplication(sys.argv)
    window = App(builder)
    window.tabs['Scene Setup'].set_update_callback(
        lambda fig, ax : plot_setup(fig, ax, scene)
        )
    window.tabs['Trajectory'].set_update_callback(
        lambda fig, ax : plot_star_trajectory_on_canvas(fig, ax, scene)
        )
    p = fluxmap_plotter()
    window.tabs['Simulation'].set_update_callback(
        lambda fig, ax : p(fig, ax, scene)
        )
    window.tabs['Simulation'].set_timer()
    window.tabs['Light Spectrum'].set_update_callback(
        lambda fig, ax : plot_spectrum(fig, ax, scene, range=(1e8,0.5e16))
    )
    sys.exit(app.exec())
