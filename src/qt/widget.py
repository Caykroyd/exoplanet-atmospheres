from PyQt5.QtWidgets import QWidget, QLineEdit, QSlider, QHBoxLayout
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

class FloatField(EditBox):
    def __init__(self, validator_range, getter, setter, callback):
        super().__init__(DoubleValidator(*validator_range), float, getter, setter, callback)

class RangeField(WidgetGroup):
    def __init__(self, validator_range, getter, setter, callback, scale = 1):
        self.scale = scale
        field0 = FloatField(validator_range, *self.single_property(getter, setter, 0), callback)
        field1 = FloatField(validator_range, *self.single_property(getter, setter, 1), callback)
        super().__init__(QHBoxLayout(),[field0, field1])

    def single_property(self, getter, setter, i):
        scale = self.scale
        _val = getter()
        print('Got',_val)
        def getter_wrapper(getter, i):
            _val = getter()
            return _val[i]*scale
        def setter_wrapper(x, setter, i):
            _val[i] = x/scale
            setter(_val)
            print(_val)
        getter_i = lambda : getter_wrapper(getter, i)
        setter_i = lambda val : setter_wrapper(val, setter, i)
        return getter_i, setter_i

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
