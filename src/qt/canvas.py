import matplotlib
matplotlib.use('Qt5Agg')

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

class PyplotCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100, proj=None):
        self.proj = proj
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111, projection=self.proj)
        self.callback = lambda fig, ax : None
        super().__init__(self.fig)
        self.setMinimumSize(800,600)

    def refresh(self):
        self.ax.clear()
        self.callback(self.fig, self.ax)
        self.fig.canvas.draw()
