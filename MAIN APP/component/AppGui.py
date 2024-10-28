import sys
import master_class as mc
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton,
                              QFileDialog, QLabel, QTextEdit, QHBoxLayout, 
                              QGridLayout, QLayout, QDialog, QListWidget, 
                              QLineEdit, QGroupBox, QSlider)
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtCore import Qt, QEvent
import os
import numpy as np
import styles
from components import ComboBox, MyBar
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from io import BytesIO
from copy import copy

class InputDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('New File-Name')
        self.setGeometry(200, 200, 300, 100)

        layout = QVBoxLayout()

        self.label = QLabel('Please enter a File-Name:', self)
        layout.addWidget(self.label)

        self.input_field = QLineEdit(self)
        layout.addWidget(self.input_field)

        self.submit_button = QPushButton('Submit', self)
        self.submit_button.clicked.connect(self.submit)
        layout.addWidget(self.submit_button)

        self.setLayout(layout)
        self.input_text = None

    def submit(self):
        self.input_text = self.input_field.text()
        self.accept()

class FileSelectionWindow(QDialog):
    def __init__(self, file_names, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Select a File')
        self.setGeometry(150, 150, 400, 300)

        layout = QVBoxLayout()

        self.list_widget = QListWidget()
        self.list_widget.addItems(file_names)
        layout.addWidget(self.list_widget)

        self.select_button = QPushButton('Select')
        self.select_button.clicked.connect(self.select_file)
        layout.addWidget(self.select_button)

        self.setLayout(layout)
        self.selected_file = None

    def select_file(self):
        selected_items = self.list_widget.selectedItems()
        if selected_items:
            self.selected_file = selected_items[0].text()
            self.accept()


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
        self.btn_warp.clicked.connect(self.PerformeWarping)
        self.btn_warp.setEnabled(False)
        InputLayout.addWidget(self.btn_warp, 4, 0)

        self.btn_plot = QPushButton('Plot Chromatogram', self)
        self.btn_plot.clicked.connect(self.PlotWarpedChromatograms)
        self.btn_plot.setEnabled(False)
        InputLayout.addWidget(self.btn_plot, 5, 0)


        # Buttons in die erste Spalte
        for i in range(6, 8):
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
        self.slider1.setMinimum(0)
        self.slider1.setMaximum(20)
        self.slider1.setValue(10)
        self.slider1.setTickPosition(QSlider.TicksBelow)
        self.slider1.setTickInterval(2)
        self.slider1.setSingleStep(1)
        self.slider1_min = QLabel('0')
        self.slider1_max = QLabel('100')
        ParameterLayout.addWidget(QLabel(parameter1+(20-len(parameter1))*' '), 0, 0)
        self.slider1_value_label = QLabel(str(self.slider1.value()))
        self.slider1.valueChanged.connect(lambda: self.slider1_value_label.setText(str(self.slider1.value())))
        ParameterLayout.addWidget(self.slider1_value_label, 0, 1)
        ParameterLayout.addWidget(self.slider1_min, 0, 2)
        ParameterLayout.addWidget(self.slider1, 0, 3)
        ParameterLayout.addWidget(self.slider1_max, 0, 4)
        

        # Slider for the second parameter
        parameter2 = 'Segment Length'
        self.slider2 = QSlider(Qt.Horizontal)
        self.slider2.setMinimum(0)
        self.slider2.setMaximum(100)
        self.slider2.setValue(50)
        self.slider2.setTickPosition(QSlider.TicksBelow)
        self.slider2.setTickInterval(10)
        self.slider2.setSingleStep(1)
        self.slider2_min = QLabel('0')
        self.slider2_max = QLabel('100')
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


        # Bildfelder in die zweite und dritte Reihe, zweite Spalte
        self.image_top = QLabel('Bildfeld 1', self)
        self.image_top.setPixmap(QPixmap())  # Bild kann später hinzugefügt werden
        self.image_top.setStyleSheet("border: 1px solid black")
        self.image_top.setAlignment(Qt.AlignCenter)
        PlotLayout.addWidget(self.image_top)  # Erste Bild in Zeile 4

        self.image_low = QLabel('Bildfeld 2', self)
        self.image_low.setPixmap(QPixmap())  # Bild kann später hinzugefügt werden
        self.image_low.setStyleSheet("border: 1px solid black")
        self.image_low.setAlignment(Qt.AlignCenter)
        PlotLayout.addWidget(self.image_low)  # Zweite Bild in Zeile 5

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

        

    def PerformeWarping(self) -> None:
        file_names = self.data_preparation.get_file_names()
        file_names.append('All')
        dialog = FileSelectionWindow(file_names, self)
        if dialog.exec_():
            selected_target = dialog.selected_file
            if selected_target == 'All':
                selected_target = self.data_preparation.get_file_names()
        else:
            self.print_to_output('Please select a file to compare with.')
            return
        
        # get slack and segment length from sliders
        slack = self.slider1.value()
        segment_length = self.slider2.value()


        if self.selected_reference_file and selected_target:
            reference = self.data_preparation.get_chromatogram(self.selected_reference_file)
            self.warped_chromatograms = {}
            self.warp_paths = {}
            if isinstance(selected_target, str):
                selected_target = [selected_target]
            for file in selected_target:
                if file != self.selected_reference_file:
                    target = self.data_preparation.get_chromatogram(file)
                    warped_target, warp_path = mc.COW(reference, target, slack=slack, segments=segment_length)
                    self.warped_chromatograms[file] = warped_target
                    self.warp_paths[file] = warp_path
                    self.print_to_output(f'Warped {file} against Reference {self.selected_reference_file}.')
                else:
                    self.warped_chromatograms[file] = reference
        elif selected_target:
            self.print_to_output('Please select a reference file.')
        elif self.selected_reference_file:
            self.print_to_output('Please select a file to compare with.')
        else:
            self.print_to_output('Please select a file to compare with and a reference file.')

        self.btn_plot.setEnabled(True)

    
    # Plotting the chromatograms all unwarped chromatograms in the top image and all warped chromatograms in the lower image
    

    def PlotWarpedChromatograms(self) -> None:
        import matplotlib.pyplot as plt

        if not self.warped_chromatograms:
            self.print_to_output('No warped chromatograms to plot.')
            return
        rt = self.data_preparation.get_retention_time()
        def plot_chromatograms(chromatograms, title):
            fig, ax = plt.subplots(figsize=(10, 5))
            for _, chromatogram in chromatograms.items():
                ax.plot(rt, chromatogram)
                ax.set_title(title)
                ax.set_xlabel('Retention Time')
                ax.set_ylabel('Intensity')
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)

            buf = BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            pixmap = QPixmap()
            pixmap.loadFromData(buf.getvalue())
            plt.close(fig)
            return pixmap

        self.image_top.setPixmap(plot_chromatograms(self.chromatograms, 'Unwarped Chromatograms'))
        self.image_low.setPixmap(plot_chromatograms(self.warped_chromatograms, 'Warped Chromatograms'))

        self.print_to_output('Chromatograms plotted.')



if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainWindow()
    ex.show()
    sys.exit(app.exec_())