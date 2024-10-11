import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QTextEdit, QComboBox, QSystemTrayIcon, QMenu, QAction, QMainWindow, QMessageBox
from PyQt5.QtGui import QIcon

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from master_class import DataPreparation  # Import der Klasse


class PlotCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(PlotCanvas, self).__init__(fig)
        self.setParent(parent)

    def plot(self, data):
        self.axes.clear()
        self.axes.plot(data)
        self.draw()


class FolderSelector(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Levins MS Project')
        self.setGeometry(300, 300, 300, 200)

        # Haupt-Widget und Layout
        self.main_widget = QWidget(self)
        layout = QVBoxLayout(self.main_widget)
        

        self.label = QLabel('Kein Ordner ausgewählt', self)
        layout.addWidget(self.label)

        self.btn_select = QPushButton('Ordner auswählen', self)
        self.btn_select.clicked.connect(self.showDialog)
        layout.addWidget(self.btn_select)

        self.btn_initialize = QPushButton('Initialize DataPreparation', self)
        self.btn_initialize.clicked.connect(self.initializeDataPreparation)
        self.btn_initialize.setEnabled(False)  # Deaktivieren, bis ein Ordner ausgewählt wurde
        layout.addWidget(self.btn_initialize)

        self.btn_show_file_in_folder = QPushButton('Check for Files',self)
        self.btn_show_file_in_folder.clicked.connect(self.show_list_list_of_existing_files)
        layout.addWidget(self.btn_show_file_in_folder)

        # QTextEdit Feld für die Ausgabe
        self.output_field = QTextEdit(self)
        self.output_field.setReadOnly(True)  # Nur Lesezugriff
        layout.addWidget(self.output_field)

        # Plot Canvas hinzufügen
        self.plot_canvas = PlotCanvas(self, width=5, height=4)
        layout.addWidget(self.plot_canvas)

        self.setLayout(layout)
        self.selected_folder = None
        self.DataPrepClass = None

    def showDialog(self):
        # Öffnet einen Dialog zum Auswählen eines Ordners
        folder_path = QFileDialog.getExistingDirectory(self, 'Ordner auswählen')

        if folder_path:
            self.selected_folder = folder_path
            self.label.setText(f'Gewählter Ordner: {folder_path}')
            self.btn_initialize.setEnabled(True)  # Aktivieren, wenn ein Ordner ausgewählt wurde
        else:
            self.label.setText('Kein Ordner ausgewählt')

    def initializeDataPreparation(self):
        if self.selected_folder:
            self.DataPrepClass = DataPreparation(self.selected_folder)
            self.print_to_output(f'DataPreparation initialized with folder: {self.selected_folder}')


        # Dummy-Daten plotten
        data = np.random.rand(10)
        self.plot_canvas.plot(data)  # Die plot-Methode von PlotCanvas aufrufen



    def print_to_output(self, text):
        self.output_field.append(text)  # Fügt Text am Ende des QTextEdit hinzu

    def show_list_list_of_existing_files(self):
        list_of_files = self.DataPrepClass.get_name_mzml_files()

        if list_of_files:
            self.output_field.append('Folder '+ self.selected_folder +' contain files:')
            for i in list_of_files:
                self.output_field.append(i)

    



if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = FolderSelector()
    ex.show()
    sys.exit(app.exec_())
