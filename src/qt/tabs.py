from PyQt5 import QtCore, QtWidgets

from PyQt5.QtWidgets import QPushButton, QWidget, QAction, QTabWidget, QVBoxLayout
from PyQt5.QtCore import pyqtSlot

from qt.widget import WidgetGroup

class Tabbar(QWidget):
    def __init__(self, tabs, parent):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)

        # Initialize the bar
        self.tabbar = QTabWidget()
        self.tabbar.resize(300,200)
        self.tabbar.currentChanged.connect(self.refresh)

        # Add tabs
        self.tabs = tabs
        for tab in tabs:
            self.tabbar.addTab(tab, tab.name)

        self.layout.addWidget(self.tabbar)
        self.setLayout(self.layout)

    def __getitem__(self, key):
        single, = (t for t in self.tabs if t.name == key)
        return single

    @pyqtSlot()
    def on_click(self):
        print("\n")
        for currentQTableWidgetItem in self.tableWidget.selectedItems():
            print(currentQTableWidgetItem.row(), currentQTableWidgetItem.column(), currentQTableWidgetItem.text())

    def refresh(self):
        self.tabs[self.tabbar.currentIndex()].refresh()

class Tab(WidgetGroup):
    def __init__(self, name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name
