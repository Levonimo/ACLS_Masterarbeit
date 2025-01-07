import sys
import os
import numpy as np
from copy import copy
import matplotlib.pyplot as plt

from . import master_class as mc
from . import styles
from .Warping import correlation_optimized_warping as COW
from .styles_pyqtgraph import graph_style_chromatogram

from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton,
                              QFileDialog, QLabel, QTextEdit,  
                              QGridLayout, QGroupBox, QSlider)
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtCore import Qt

import pyqtgraph as pg

from .ExternalGui import InputDialog, FileSelectionWindow, PCAWindow
from .components import MyBar
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas





class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.chromatograms = dict()
        
        # Set window title and size
        self.setWindowTitle('GC-MS Warping Tool')
        # Set the window icon
        icon_path = os.path.join(os.path.dirname(__file__), 'images', 'Logo_ICBT_Analytik_round.ico')
        app_icon = QIcon(icon_path)
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
        self.btn_select_file.clicked.connect(self.SelectionReferenceFile)
        self.btn_select_file.setEnabled(False)
        InputLayout.addWidget(self.btn_select_file, 3, 0)

        self.btn_warp = QPushButton('Performe Warping', self)
        self.btn_warp.clicked.connect(self.PerformeWarping)
        self.btn_warp.setEnabled(False)
        InputLayout.addWidget(self.btn_warp, 4, 0)

        self.btn_plot = QPushButton('Plot Chromatogram', self)
        self.btn_plot.clicked.connect(self.PlotWarpedChromatograms)
        self.btn_plot.setEnabled(False)
        InputLayout.addWidget(self.btn_plot, 5, 0)

        self.btn_analyse = QPushButton('Statistic Analyse', self)
        self.btn_analyse.clicked.connect(self.StatisticAnalyse)
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


        ParameterGroupBox = QGroupBox("Adjust Parameter", objectName="Init")

        ParameterLayout = QGridLayout()

        # Added Parameter selection for the warping tool as sliders
        # Slider for the first parameter
        parameter1 = 'Slack'
        self.slider1 = QSlider(Qt.Horizontal)
        self.slider1.setMinimum(1)
        self.slider1.setMaximum(40)
        self.slider1.setValue(14)
        self.slider1.setTickPosition(QSlider.TicksBelow)
        self.slider1.setTickInterval(2)
        self.slider1.setSingleStep(1)
        self.slider1_min = QLabel('1')
        self.slider1_max = QLabel('40')
        ParameterLayout.addWidget(QLabel(parameter1+(20-len(parameter1))*' '), 0, 0)
        self.slider1_value_label = QLabel(str(self.slider1.value()))
        self.slider1.valueChanged.connect(lambda: self.slider1_value_label.setText(str(self.slider1.value())))
        ParameterLayout.addWidget(self.slider1_value_label, 0, 1)
        ParameterLayout.addWidget(self.slider1_min, 0, 2)
        ParameterLayout.addWidget(self.slider1, 0, 3)
        ParameterLayout.addWidget(self.slider1_max, 0, 4)
        

        # Slider for the second parameter
        parameter2 = 'Segments'
        self.slider2 = QSlider(Qt.Horizontal)
        self.slider2.setMinimum(10)
        self.slider2.setMaximum(300)
        self.slider2.setValue(60)
        self.slider2.setTickPosition(QSlider.TicksBelow)
        self.slider2.setTickInterval(20)
        self.slider2.setSingleStep(5)
        self.slider2_min = QLabel('10')
        self.slider2_max = QLabel('1000')
        ParameterLayout.addWidget(QLabel(parameter2+(20-len(parameter2))*' '), 1, 0)
        self.slider2_value_label = QLabel(str(self.slider2.value()))
        self.slider2.valueChanged.connect(lambda: self.slider2_value_label.setText(str(self.slider2.value())))
        ParameterLayout.addWidget(self.slider2_value_label, 1, 1)
        ParameterLayout.addWidget(self.slider2_min, 1, 2)
        ParameterLayout.addWidget(self.slider2, 1, 3)
        ParameterLayout.addWidget(self.slider2_max, 1, 4)


        ParameterGroupBox.setFixedHeight(100)
        ParameterGroupBox.setLayout(ParameterLayout)
        MainWindow.addWidget(ParameterGroupBox)




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
            self.data_preparation = mc.DataPreparation(self.selected_folder)
            self.print_to_output(f'DataPreparation initialized with folder: {self.selected_folder}')
            self.npy_import()
            if self.chromatograms:
                self.btn_show_files.setEnabled(True)
                self.btn_select_file.setEnabled(True)

            self.rt = self.data_preparation.get_retention_time()
            np.save('./Outputs/retention_time.npy', self.rt)
          


    def print_to_output(self, text: str) -> None:
        self.output_field.append(text)  # Fügt Text am Ende des QTextEdit hinzu

    def ShowNameOfAllFiles(self) -> None:
        if self.data_preparation:
            self.print_to_output('Files in selected folder:')
            file_names = self.data_preparation.get_file_names()
            for i in range(0, len(file_names), 4):
                self.print_to_output(' | '.join(file_names[i:i + 4]))

    def SelectionReferenceFile(self) -> None:
        if self.data_preparation:
            file_names = self.data_preparation.get_file_names()
            dialog = FileSelectionWindow(file_names, self)
            if dialog.exec_():
                self.selected_reference_file = copy(dialog.selected_file)
                self.print_to_output(f'Reference file: {self.selected_reference_file}')
                self.plot_graph_top.clear()
                self.plot_graph_top.plot(self.rt, np.sum(self.chromatograms[self.selected_reference_file], axis=1), pen=pg.mkPen(color=(0, 0, 0)))
                self.plot_graph_top.setTitle('Unwarped Chromatograms')
                self.plot_graph_top.setLabel('left', 'Intensity')
                self.plot_graph_top.setLabel('bottom', 'Retention Time')
        self.btn_warp.setEnabled(True)

    def npy_import(self) -> None:
        if self.selected_folder:
            npy_files = [file for file in os.listdir(self.selected_folder) if file.endswith('.npy')]
            if npy_files:
                npy_files.append('<New File>')
                dialog = FileSelectionWindow(npy_files, self)
                if dialog.exec_():
                    selected_Chromatograms = dialog.selected_file
                    if selected_Chromatograms == '<New File>':
                        input_dialog = InputDialog(self)
                        if input_dialog.exec_():
                            input_word = input_dialog.input_text
                            self.print_to_output(f'New File Named: {input_word}.npy')
                            self.chromatograms = self.data_preparation.get_list_of_chromatograms(input_word, file_list=self.data_preparation.get_file_names())
                    else:
                        self.print_to_output(f'Chromatograms from {selected_Chromatograms} loaded.')
                        self.chromatograms = self.data_preparation.get_list_of_chromatograms(selected_Chromatograms)
                
                    
            else:
                input_dialog = InputDialog(self)
                if input_dialog.exec_():
                    input_word = input_dialog.input_text
                    self.print_to_output(f'New File Named: {input_word}.npy')

                    self.chromatograms = self.data_preparation.get_list_of_chromatograms(input_word, file_list=self.data_preparation.get_file_names())
        
        # check if chromatograms have the same length if not short them to the same length, take away the first elements
        uniq_len = np.unique([len(chromatogram) for chromatogram in self.chromatograms.values()])
        if uniq_len.size > 1:
            min_len = np.min(uniq_len)
            for name, chromatogram in self.chromatograms.items():
                self.chromatograms[name] = chromatogram[-min_len:]

        

    def PerformeWarping(self) -> None:
        file_names = self.data_preparation.get_file_names()
        self.selected_target = None
        # add 'All' at the beginning to the list of file names
        file_names.insert(0, 'All')

        dialog = FileSelectionWindow(file_names, self)
        if dialog.exec_():
            self.selected_target = dialog.selected_file
            if self.selected_target == 'All':
                self.selected_target = self.data_preparation.get_file_names()
        else:
            self.print_to_output('Please select a file to compare with.')
            return
        
        # get slack and segment length from sliders
        slack = self.slider1.value()
        segment_length = self.slider2.value()


        if self.selected_reference_file and self.selected_target:
            reference = self.chromatograms[self.selected_reference_file]
            self.warped_chromatograms = {}
            self.warp_paths = {}
            self.similarity_scores = {}
            if isinstance(self.selected_target, str):
                self.selected_target = [self.selected_target]
            for target_file in self.selected_target:
                if target_file != self.selected_reference_file:
                    target = self.chromatograms[target_file]
                    warped_target, warp_path, score = COW(reference_2D = reference, target_2D = target, slack=slack, segments=segment_length)
                    self.warped_chromatograms[target_file] = warped_target
                    self.warp_paths[target_file] = warp_path
                    self.similarity_scores[target_file] = score
                    self.print_to_output(f'Warped {target_file} against Reference {self.selected_reference_file}.')
                else:
                    self.warped_chromatograms[target_file] = reference
        elif self.selected_target:
            self.print_to_output('Please select a reference file.')
        elif self.selected_reference_file:
            self.print_to_output('Please select a file to compare with.')
        else:
            self.print_to_output('Please select a file to compare with and a reference file.')

        self.btn_plot.setEnabled(True)
        self.btn_analyse.setEnabled(True)
        
        # np.save('./Outputs/warped_chromatograms.npy', self.warped_chromatograms)
        # np.save('./Outputs/unwarped_chromatograms.npy', self.chromatograms)
        # np.save('./Outputs/selected_target.npy', self.selected_target)
        
    
    # Plotting the chromatograms all unwarped chromatograms in the top image and all warped chromatograms in the lower image
    

    def PlotWarpedChromatograms(self) -> None:
        if not self.warped_chromatograms:
            self.print_to_output('No warped chromatograms to plot.')
            return
        
        self.plot_graph_top.clear()
        self.plot_graph_bottom.clear()
        
        self.plot_graph_top.plot(self.rt, np.sum(self.chromatograms[self.selected_reference_file], axis=1), pen=pg.mkPen(color=(0, 0, 0)))
        for name, chromatogram in self.chromatograms.items():
            if name in self.selected_target:
                self.plot_graph_top.plot(self.rt, np.sum(chromatogram, axis=1))
        self.plot_graph_top.setTitle('Unwarped Chromatograms')
        self.plot_graph_top.setLabel('left', 'Intensity')
        self.plot_graph_top.setLabel('bottom', 'Retention Time')

        self.plot_graph_bottom.plot(self.rt, np.sum(self.chromatograms[self.selected_reference_file], axis=1), pen=pg.mkPen(color=(0, 0, 0)))
        for _, chromatogram in self.warped_chromatograms.items():
            self.plot_graph_bottom.plot(self.rt, np.sum(chromatogram, axis=1))
        self.plot_graph_bottom.setTitle('Warped Chromatograms')
        self.plot_graph_bottom.setLabel('left', 'Intensity')
        self.plot_graph_bottom.setLabel('bottom', 'Retention Time')


        self.print_to_output('Chromatograms plotted.')

    def StatisticAnalyse(self) -> None:
        dialog = PCAWindow(self.selected_target,self.warped_chromatograms,self.chromatograms, self.rt, self)
        if dialog.exec_():
            results = dialog.results
            self.print_to_output('PCA finished.')
            
            #plot the loadings in bottom plot
            self.plot_graph_bottom.clear()
            # if ddimension of loadings is more than 1, sum it up
            if len(results['loadings']) > 1:
                self.plot_graph_bottom.plot(self.rt , np.sum(results['loadings'], axis=1), pen=pg.mkPen(color=(0, 0, 0)))
            else:
                self.plot_graph_bottom.plot(self.rt , results['loadings'][0], pen=pg.mkPen(color=(0, 0, 0)))
            self.plot_graph_bottom.setLabel('left', 'Intensity')
            self.plot_graph_bottom.setLabel('bottom', 'Retention Time')
        



# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     ex = MainWindow()
#     ex.show()
#     sys.exit(app.exec_())