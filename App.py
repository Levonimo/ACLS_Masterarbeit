import sys
import master_class as mc
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QTextEdit, QHBoxLayout, QGridLayout
from PyQt5.QtWidgets import QDialog, QListWidget, QLineEdit
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import os
import numpy as np

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
        self.initUI()

    def initUI(self):
        # Set window title and size
        self.setWindowTitle('GC-MS Warping Tool')
        self.setGeometry(100, 100, 800, 600)

        # Hauptlayout - ein Gitterlayout
        grid = QGridLayout()
        self.setLayout(grid)

        # Buttons in die erste Spalte
        self.btn_select = QPushButton('Ordner auswählen', self)
        self.btn_select.clicked.connect(self.selectFolder)
        grid.addWidget(self.btn_select, 0, 0)

        self.btn_init = QPushButton('Initialize DataPreparation', self)
        self.btn_init.clicked.connect(self.initializeDataPreparation)
        self.btn_init.setEnabled(False)
        grid.addWidget(self.btn_init, 1, 0)

        self.btn_show_files = QPushButton('Show Name of all Files', self)
        self.btn_show_files.clicked.connect(self.ShowNameOfAllFiles)
        self.btn_show_files.setEnabled(False)
        grid.addWidget(self.btn_show_files, 2, 0)

        self.btn_select_file = QPushButton('Select a File', self)
        self.btn_select_file.clicked.connect(self.openFileSelectionWindow)
        self.btn_select_file.setEnabled(False)
        grid.addWidget(self.btn_select_file, 3, 0)

        self.btn_warp = QPushButton('Performe Warping', self)
        self.btn_warp.clicked.connect(self.PerformeWarping)
        self.btn_warp.setEnabled(False)
        grid.addWidget(self.btn_warp, 4, 0)

        # Buttons in die erste Spalte
        for i in range(5, 8):
            button = QPushButton(f'Button {i + 1}', self)
            grid.addWidget(button, i, 0)  # Positioniere die Buttons in Spalte 0, Zeilen 0-7

        # Textfeld in die erste Reihe, zweite Spalte
        self.output_field = QTextEdit(self)
        self.output_field.setPlainText("Waiting for input...")
        self.output_field.setReadOnly(True)
        grid.addWidget(self.output_field, 0, 1, 8, 1)  # Textfeld über 4 Zeilen

        # Bildfelder in die zweite und dritte Reihe, zweite Spalte
        image_top = QLabel('Bildfeld 1', self)
        image_top.setPixmap(QPixmap())  # Bild kann später hinzugefügt werden
        image_top.setStyleSheet("border: 1px solid black")
        image_top.setAlignment(Qt.AlignCenter)
        grid.addWidget(image_top, 8, 0, 2, 3)  # Erste Bild in Zeile 4

        image_low = QLabel('Bildfeld 2', self)
        image_low.setPixmap(QPixmap())  # Bild kann später hinzugefügt werden
        image_low.setStyleSheet("border: 1px solid black")
        image_low.setAlignment(Qt.AlignCenter)
        grid.addWidget(image_low, 10, 0, 2, 3)  # Zweite Bild in Zeile 5

        # Layout-Anpassungen für die Spaltenbreite
        grid.setColumnStretch(0, 1)  # Buttons Spalte
        grid.setColumnStretch(1, 2)  # Textfeld und Bildfelder Spalte


        self.selected_folder = None
        self.DataPrepClass = None

    def selectFolder(self):
        # Öffnet einen Dialog zum Auswählen eines Ordners
        folder_path = QFileDialog.getExistingDirectory(self, 'Ordner auswählen')

        if folder_path:
            self.selected_folder = folder_path
            self.print_to_output(f'Gewählter Ordner: {folder_path}')
            self.btn_init.setEnabled(True)  # Aktivieren, wenn ein Ordner ausgewählt wurde
        else:
            self.print_to_output('Kein Ordner ausgewählt')


    def initializeDataPreparation(self):
        if self.selected_folder:
            self.data_preparation = mc.DataPreparation(self.selected_folder)
            self.print_to_output(f'DataPreparation initialized with folder: {self.selected_folder}')
            self.npy_import()
            if self.chromatograms:
                self.btn_show_files.setEnabled(True)
                self.btn_select_file.setEnabled(True)
                self.btn_warp.setEnabled(True)


    def print_to_output(self, text):
        self.output_field.append(text)  # Fügt Text am Ende des QTextEdit hinzu

    def ShowNameOfAllFiles(self):
        if self.data_preparation:
            self.print_to_output('Files in selected folder:')
            file_names = self.data_preparation.get_file_names()
            for i in range(0, len(file_names), 4):
                self.print_to_output(' | '.join(file_names[i:i + 4]))

    def openFileSelectionWindow(self):
        if self.data_preparation:
            file_names = self.data_preparation.get_file_names()
            dialog = FileSelectionWindow(file_names, self)
            if dialog.exec_():
                self.selected_reference_file = dialog.selected_file
                self.print_to_output(f'Reference file: {self.selected_reference_file}')

    def npy_import(self):
        if self.selected_folder:
            npy_files = [file for file in os.listdir(self.selected_folder) if file.endswith('.npy')]
            if npy_files:
                dialog = FileSelectionWindow(npy_files, self)
                if dialog.exec_():
                    selected_Chromatograms = dialog.selected_file
                    self.print_to_output(f'Chromatograms from {selected_Chromatograms} loaded.')
                    self.chromatograms = np.load(self.selected_folder + '/' + selected_Chromatograms, allow_pickle=True).item()
            else:
                input_dialog = InputDialog(self)
                if input_dialog.exec_():
                    input_word = input_dialog.input_text
                    self.print_to_output(f'New File Named: {input_word}.npy')

                    self.chromatograms = self.data_preparation.get_list_of_chromatograms(input_word, file_list=self.data_preparation.get_file_names())

    def PerformeWarping(self):
        file_names = self.data_preparation.get_file_names().append('All')
        dialog = FileSelectionWindow(file_names, self)
        if dialog.exec_():
            selected_target = dialog.selected_file
            if selected_target == 'All':
                selected_target = self.data_preparation.get_file_names()
        else:
            self.print_to_output('Please select a file to compare with.')
            return

        if self.selected_reference_file and selected_target:
            reference = self.data_preparation.get_chromatogram(self.selected_reference_file)
            self.warped_chromatograms = {}
            for file in selected_target:
                if file != self.selected_reference_file:
                    target = self.data_preparation.get_chromatogram(file)
                    warped_target, warp_path = mc.COW(reference, target)
                    self.warped_chromatograms[file] = warped_target
                    self.print_to_output(f'Warping Path for {file}: {warp_path}')
                else:
                    self.warped_chromatograms[file] = reference
        elif selected_target:
            self.print_to_output('Please select a reference file.')
        elif self.selected_reference_file:
            self.print_to_output('Please select a file to compare with.')
        else:
            self.print_to_output('Please select a file to compare with and a reference file.')



if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainWindow()
    ex.show()
    sys.exit(app.exec_())