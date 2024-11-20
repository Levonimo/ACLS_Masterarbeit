import sys
import os
import numpy as np
from copy import copy


from . import styles
from .styles_pyqtgraph import graph_style_chromatogram

from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton,
                              QFileDialog, QLabel, QTextEdit,  
                              QGridLayout, QGroupBox, QSlider)
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtCore import Qt

import pyqtgraph as pg

from .components import MyBar
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas





class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.chromatograms = dict()
        
        # Set window title and size
        self.setWindowTitle('GC-MS Warping Tool')
        # Set the window icon
        app_icon = QIcon("images/Logo_ICBT_Analytik_round.ico")
        self.setWindowIcon(app_icon)

        # Layout
        self.setStyleSheet(styles.Levin)
        self.setMinimumSize(900, 1200)
        self.setWindowFlags(Qt.FramelessWindowHint)

        # Hauptlayout - ein Gitterlayout
        MainWindow = QGridLayout()
        self.setLayout(MainWindow)

        # Added MenuBar from Nicolas Imstepf
        self.MenuBar = MyBar(self)
        self.MenuBar.setFixedHeight(40)
        MainWindow.addWidget(self.MenuBar)
        

        InputGroupBox = QGroupBox("Init", objectName="Init")
        InputLayout = QGridLayout()
        
        # Buttons in die erste Spalte
        self.btn_select = QPushButton('Ordner auswählen', self)
        self.btn_select.clicked.connect(self.selectFolder)
        InputLayout.addWidget(self.btn_select, 0, 0)

        self.btn_init = QPushButton('Initialize DataPreparation', self)
        self.btn_init.clicked.connect(self.initializeDataPreparation)
        self.btn_init.setEnabled(False)
        InputLayout.addWidget(self.btn_init, 1, 0)

        self.btn_show_files = QPushButton('Show Name of all Files', self)
        self.btn_show_files.clicked.connect(self.ShowNameOfAllFiles)
        self.btn_show_files.setEnabled(False)
        InputLayout.addWidget(self.btn_show_files, 2, 0)

        self.btn_select_file = QPushButton('Select Reference File', self)
        self.btn_select_file.clicked.connect(self.openFileSelectionWindow)
        self.btn_select_file.setEnabled(False)
        InputLayout.addWidget(self.btn_select_file, 3, 0)

        self.btn_warp = QPushButton('Performe Warping', self)
        #self.btn_warp.clicked.connect()
        self.btn_warp.setEnabled(False)
        InputLayout.addWidget(self.btn_warp, 4, 0)

        self.btn_plot = QPushButton('Plot Chromatogram', self)
        #self.btn_plot.clicked.connect()
        self.btn_plot.setEnabled(False)
        InputLayout.addWidget(self.btn_plot, 5, 0)

        self.btn_analyse = QPushButton('Statistic Analyse', self)
        #self.btn_analyse.clicked.connect( )
        self.btn_analyse.setEnabled(False)
        InputLayout.addWidget(self.btn_analyse, 6, 0)


        # Buttons in die erste Spalte
        for i in range(7, 8):
            button = QPushButton(f'Button {i + 1}', self)
            InputLayout.addWidget(button, i, 0)  # Positioniere die Buttons in Spalte 0, Zeilen 0-7

        # Textfeld in die erste Reihe, zweite Spalte
        self.output_field = QTextEdit(self)
        self.output_field.setPlainText("Waiting for input...")
        self.output_field.setReadOnly(True)
        InputLayout.addWidget(self.output_field, 0, 1, 8, 1)  # Textfeld über 8 Zeilen

        InputGroupBox.setLayout(InputLayout)
        InputGroupBox.setFixedHeight(250)
        MainWindow.addWidget(InputGroupBox)


        PlotGroupBox = QGroupBox("Plots", objectName="Init")
        PlotLayout = QGridLayout()


        # Plot Windows for the chromatograms with pyqtgraph
        self.plot_graph_top = pg.PlotWidget()
        graph_style_chromatogram(self.plot_graph_top)
        PlotLayout.addWidget(self.plot_graph_top)

        self.plot_graph_bottom = pg.PlotWidget()
        graph_style_chromatogram(self.plot_graph_bottom)
        PlotLayout.addWidget(self.plot_graph_bottom)

        PlotGroupBox.setLayout(PlotLayout)
        MainWindow.addWidget(PlotGroupBox)

        # Layout-Anpassungen für die Spaltenbreite
        #MainWindow.setColumnStretch(0, 1)  # Buttons Spalte
        #MainWindow.setColumnStretch(1, 2)  # Textfeld und Bildfelder Spalte


        self.selected_folder = None
        self.DataPrepClass = None


    
    def selectFolder(self) -> None:
        # Öffnet einen Dialog zum Auswählen eines Ordners
        folder_path = QFileDialog.getExistingDirectory(self, 'Ordner auswählen')

        if folder_path:
            self.selected_folder = folder_path
            self.print_to_output(f'Gewählter Ordner: {folder_path}')
            self.btn_init.setEnabled(True)  # Aktivieren, wenn ein Ordner ausgewählt wurde
        else:
            self.print_to_output('Kein Ordner ausgewählt')


    def initializeDataPreparation(self) -> None:
        if self.selected_folder:
            self.print_to_output(f'DataPreparation initialized with folder: {self.selected_folder}')
            self.npy_import()
            if self.chromatograms:
                self.btn_show_files.setEnabled(True)
                self.btn_select_file.setEnabled(True)
                


    def print_to_output(self, text: str) -> None:
        self.output_field.append(text)  # Fügt Text am Ende des QTextEdit hinzu

    def ShowNameOfAllFiles(self) -> None:
        if self.data_preparation:
            self.print_to_output('Files in selected folder:')
            file_names = self.data_preparation.get_file_names()
            for i in range(0, len(file_names), 4):
                self.print_to_output(' | '.join(file_names[i:i + 4]))

    def openFileSelectionWindow(self) -> None:
        if self.data_preparation:
            file_names = self.data_preparation.get_file_names()
            dialog = FileSelectionWindow(file_names, self)
            if dialog.exec_():
                self.selected_reference_file = copy(dialog.selected_file)
                self.print_to_output(f'Reference file: {self.selected_reference_file}')
        self.btn_warp.setEnabled(True)


    def PlotWarpedChromatograms(self) -> None:
        if not self.warped_chromatograms:
            self.print_to_output('No warped chromatograms to plot.')
            return
        rt = self.data_preparation.get_retention_time()
        self.plot_graph_top.clear()
        self.plot_graph_bottom.clear()
        
        self.plot_graph_top.plot(rt, np.sum(self.chromatograms[self.selected_reference_file], axis=1), pen=pg.mkPen(color=(0, 0, 0)))
        for name, chromatogram in self.chromatograms.items():
            if name in self.selected_target:
                self.plot_graph_top.plot(rt, np.sum(chromatogram, axis=1))
        self.plot_graph_top.setTitle('Unwarped Chromatograms')
        self.plot_graph_top.setLabel('left', 'Intensity')
        self.plot_graph_top.setLabel('bottom', 'Retention Time')

        self.plot_graph_bottom.plot(rt, np.sum(self.chromatograms[self.selected_reference_file], axis=1), pen=pg.mkPen(color=(0, 0, 0)))
        for _, chromatogram in self.warped_chromatograms.items():
            self.plot_graph_bottom.plot(rt, np.sum(chromatogram, axis=1))
        self.plot_graph_bottom.setTitle('Warped Chromatograms')
        self.plot_graph_bottom.setLabel('left', 'Intensity')
        self.plot_graph_bottom.setLabel('bottom', 'Retention Time')


        self.print_to_output('Chromatograms plotted.')
