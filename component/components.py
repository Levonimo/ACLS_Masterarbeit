"""
Components of the Graphical user interface
    (Personalized PyQt objects)

adapted from David Patsch's AlphaDock and Nicola Imstepf's Master Thesis 
- RightPushButton
- MyBar

components.py
"""

from . import styles, config

import webbrowser

from PyQt5.QtWidgets import (QHBoxLayout, QPushButton, QWidget, QLayout, 
                             QMenu, QAction,  QComboBox)
from PyQt5.QtCore import Qt, QPoint, QEvent
from PyQt5.QtGui import QIcon, QStandardItemModel

from functools import partial
import re

#=========================================================================================================

class CheckableComboBox(QComboBox): 
    def __init__(self, action, parent=None): 
        super(CheckableComboBox, self).__init__(parent) 
        self.view().pressed.connect(self.handleItemPressed) 
        self.setModel(QStandardItemModel(self)) 
        self.action = action

    def addItems(self, items):
        for item in items:
            self.addItem(item)
            item = self.model().item(self.count() - 1)
            item.setCheckState(Qt.Unchecked)

    def handleItemPressed(self, index): 
        item = self.model().itemFromIndex(index) 
        if item.checkState() == Qt.Checked: 
            item.setCheckState(Qt.Unchecked) 
        else: 
            item.setCheckState(Qt.Checked)
        self.action.trigger(item.text(), item.checkState() == Qt.Checked)
#=========================================================================================================
class RightPushButton(QPushButton):
    """
    QPushbutton with additional function on right click
    adapted from stackoverflow.com/questions/44264157
    """
    def __init__(self, parent: QLayout) -> None:
        super(RightPushButton, self).__init__()
        self.parent = parent

    def mousePressEvent(self, QMouseEvent: QEvent) -> None:
        """If rightclicked, call rightClickedEvent function from InteractionGUI"""
        if QMouseEvent.button() == Qt.RightButton:
            self.parent.parent.rightClickedEvent()
       
#=========================================================================================================       
class MyBar(QWidget):
    """QWidget to replace the removed title bar and buttons"""

    def __init__(self, parent: QLayout) -> None:
        super(MyBar, self).__init__()
        
        # initialize general variables
        self.setStyleSheet(styles.Levin)      
        self.parent = parent
        
        # general settings
        layout = QHBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        btn_size = 40

        # left hand side
        leftWidget = QWidget()
        leftLayout = QHBoxLayout()
        leftLayout.setContentsMargins(0,0,0,0)

        # logo
        analyticlogo = QPushButton()
        analyticlogo.setStyleSheet(styles.analyticlogo)
        analyticlogo.setFixedSize(btn_size,btn_size)
        

        # settings
        btn_settings = QPushButton()
        btn_settings.setToolTip('Settings (Ctrl+I)')
        btn_settings.setShortcut("Ctrl+I")
        btn_settings.setFixedSize(btn_size,btn_size)
        btn_settings.setStyleSheet(styles.SettingsButton)

        self.actionshowDist = QAction("interaction lengths", self)
        self.actionshowDist.setShortcut("Ctrl+D")
        self.actionshowDist.setCheckable(True)

        self.actionclean_up = QAction("reset everything", self)
        self.actionclean_up.setShortcut("Ctrl+R")

        menu_setting = QMenu()
        menu_setting.setToolTipsVisible(True)
        menu_setting.setStyleSheet(styles.Levin)
        menu_setting.addAction(self.actionclean_up)
        menu_setting.addAction(self.actionshowDist)
        btn_settings.setMenu(menu_setting)

        # help
        btn_help = QPushButton()
        btn_help.clicked.connect(lambda: webbrowser.open(config.HELP_PATH))
        btn_help.setToolTip('Help (Ctrl+H)')
        btn_help.setFixedSize(btn_size, btn_size)
        btn_help.setStyleSheet(styles.MapButton)
        btn_help.setShortcut("Ctrl+H")

        leftLayout.addWidget(analyticlogo)
        leftLayout.addWidget(btn_settings)
        leftLayout.addWidget(btn_help)
        leftWidget.setLayout(leftLayout)
        layout.addWidget(leftWidget, alignment=Qt.AlignLeft)

        # right hand side
        rightWidget = QWidget()
        rightLayout = QHBoxLayout()
        rightLayout.setContentsMargins(0,0,0,0)

        # minimize window
        btn_min = QPushButton()
        btn_min.clicked.connect(lambda: self.parent.showMinimized())
        btn_min.setFixedSize(btn_size, btn_size)
        btn_min.setStyleSheet(styles.MinButton)

        # close window
        btn_close = QPushButton()
        btn_close.clicked.connect(lambda: self.parent.close())
        btn_close.setFixedSize(btn_size,btn_size)
        btn_close.setStyleSheet(styles.CloseButton)

        rightLayout.addWidget(btn_min)
        rightLayout.addWidget(btn_close)
        rightWidget.setLayout(rightLayout)
        layout.addWidget(rightWidget, alignment=Qt.AlignRight)

        self.setLayout(layout)

        # ability to move the window
        # adapted from from stackoverflow.com/questions/44241612
        self.start = QPoint(0, 0)
        self.pressing = False

    def mousePressEvent(self, event: QEvent) -> None:
        """
        ability to move the window
        adapted from from stackoverflow.com/questions/44241612
        """
        self.start = self.mapToGlobal(event.pos())
        self.pressing = True

    def mouseMoveEvent(self, event: QEvent) -> None:
        """
        ability to move the window
        adapted from from stackoverflow.com/questions/44241612
        """
        if self.pressing:
            self.end = self.mapToGlobal(event.pos())
            self.movement = self.end-self.start
            self.parent.setGeometry(self.mapToGlobal(self.movement).x()-10,
                                self.mapToGlobal(self.movement).y()-10,
                                self.parent.width(),
                                self.parent.height())
            self.start = self.end
