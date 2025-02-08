from PyQt5.QtWidgets import (QDialog, QLabel, QLineEdit, QPushButton, 
                             QCheckBox, QComboBox,QGridLayout, QGroupBox,
                             QMessageBox, QToolTip)

from PyQt5.QtGui import QCursor, QPixmap

import sys 
import os
from .GUI_Selection import CrossrefFileSelectionWindow
from .fun_PCA import perform_pca
from .styles_pyqtgraph import graph_style_chromatogram
import pyqtgraph as pg
from .fun_Groupmaker import GroupMaker
from .GUI_components import CheckableComboBox
import numpy as np
from copy import copy

import uuid
import logging
from datetime import datetime


class StatisticalWindow(QDialog):
    def __init__(self, scores: dict, loadings , groups, parent):
        super().__init__(parent)

        self.scores = scores
        self.loadings = loadings
        self.groups = groups
        self.parent = parent

        self.setWindowTitle('PCA Settings')
        self.setMinimumSize(400, 400)
        

        layout = QGridLayout()
        self.setLayout(layout)

        
        
        self.initUI()