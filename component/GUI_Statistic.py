from PyQt5.QtWidgets import (QDialog, QLabel, QLineEdit, QPushButton, 
                             QCheckBox, QComboBox,QGridLayout, QGroupBox,
                             QMessageBox, QToolTip,QTextEdit)

from PyQt5.QtGui import QCursor, QPixmap

import sys 
import os

from .styles_pyqtgraph import graph_style_chromatogram
import pyqtgraph as pg
from .fun_Groupmaker import GroupMaker
from .GUI_components import CheckableComboBox
import numpy as np
import pandas as pd

from copy import copy

import uuid
import logging
from datetime import datetime


class StatisticalWindow(QDialog):
    def __init__(self, results, file_names, parent):
        super().__init__(parent)
        # Set Window Settings
        
        self.setWindowTitle('Statistical Analysis')
        self.setMinimumSize(400, 400)
        

        layout = QGridLayout()
        self.setLayout(layout)


        # Set window UI components
        
        # Add Box for Plotting Stuff
        PlottingGroupBox = QGroupBox("Plotting",self)
        PlottingLayout = QGridLayout(PlottingGroupBox)


        PlottingGroupBox.setLayout(PlottingLayout)
        layout.addWidget(PlottingGroupBox, 0, 0)


        # Add Box for Text Output
        OutputGroupBox = QGroupBox("Output",self)
        OutputLayout = QGridLayout(OutputGroupBox)

        self.text_output = QTextEdit(self)
        self.text_output.setReadOnly(True)
        OutputLayout.addWidget(self.text_output, 0, 0)

        OutputGroupBox.setLayout(OutputLayout)
        layout.addWidget(OutputGroupBox, 0, 1)


        # Add Box for Bottons/Parameter Selection

        SettingsGroupBox = QGroupBox("Settings",self)
        SettingsLayout = QGridLayout(SettingsGroupBox)


        SettingsGroupBox.setLayout(SettingsLayout)
        layout.addWidget(SettingsGroupBox, 0, 2)


        # Initialize the class variables

        self.results = results
        self.score_df = pd.DataFrame.from_dict(self.results['scores'], orient="index")
        self.score_df.columns = [f"PC{i+1}" for i in range(self.score_df.shape[1])]

        self.file_names = file_names

        self.Groups, self.filename_parts = GroupMaker(self.file_names)