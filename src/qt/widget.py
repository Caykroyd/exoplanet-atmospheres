
from PyQt5.QtWidgets import QWidget, QLineEdit, QSlider
from PyQt5 import QtGui
from PyQt5.QtGui import QDoubleValidator
import numpy as np

class WidgetGroup(QWidget):
    def __init__(self, layout, widgets, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layout = layout
        self.widgets = widgets

        for w in widgets:
            if w is None:
                    self.layout.addStretch()
            else:
                self.layout.addWidget(w)

        self.setLayout(layout)

class DoubleValidator(QDoubleValidator):
    def fixup(self, value):
        value = float(value)
        value = np.round(value, decimals=self.decimals())
        value = np.clip(value, self.bottom(), self.top())
        return str(value.item())

class EditBox(QLineEdit):
    '''
    setter is called when the editbox is modified
    on_changed too
    parser parses the value read from the editbox from string to whichever format (e.g. float)
    getter is called to initialise the value of the editbox
    '''

    def __init__(self, validator, parser, getter=None, setter=None, on_changed=None):
        self.getter = getter
        self.setter = setter
        self.parser = parser
        self.on_changed = on_changed

        super().__init__()
        self.setValidator(validator)
        self.setText(validator.fixup(str(getter())))
        self.textChanged.connect(self._value_changed)
        self.textChanged.emit(self.text())

    def _value_changed(self, value):
        state = self.validator().validate(self.text(), 0)[0]
        if state == QtGui.QValidator.Acceptable:
            color = '#ffffff'
            x = self.parser(value)
            self.setter(x)
            self.on_changed()
        elif state == QtGui.QValidator.Intermediate:
            color = '#fff79a' # yellow
        else:
            color = '#f6989d' # red
        self.setStyleSheet('QLineEdit {{ background-color: {} }}'.format(color)) # change colour for feedback


class Slider(QSlider):

    def __init__(self, orientation, range, getter=None, setter=None, on_changed=None):
        self.getter = getter
        self.setter = setter
        self.on_changed = on_changed

        a, b, step = range
        assert a < b and step > 0
        self.n = abs(b-a)/step

        super().__init__(orientation)
        self.setRange(a*self.n, b*self.n)
        self.setValue(float(getter()) * self.n)
        self.valueChanged.connect(self._value_changed)

    def _value_changed(self, value):
        self.setter(value / self.n)
        self.on_changed()
