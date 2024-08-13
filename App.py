import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QTextEdit
from master_class import DataPreparation  # Import der Klasse

class FolderSelector(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Ordnerauswahl')

        # Layout und Widgets
        layout = QVBoxLayout()

        self.label = QLabel('Kein Ordner ausgewählt', self)
        layout.addWidget(self.label)

        self.btn_select = QPushButton('Ordner auswählen', self)
        self.btn_select.clicked.connect(self.showDialog)
        layout.addWidget(self.btn_select)

        self.btn_initialize = QPushButton('Initialize DataPreparation', self)
        self.btn_initialize.clicked.connect(self.initializeDataPreparation)
        self.btn_initialize.setEnabled(False)  # Deaktivieren, bis ein Ordner ausgewählt wurde
        layout.addWidget(self.btn_initialize)

        # QTextEdit Feld für die Ausgabe
        self.output_field = QTextEdit(self)
        self.output_field.setReadOnly(True)  # Nur Lesezugriff
        layout.addWidget(self.output_field)

        self.setLayout(layout)
        self.selected_folder = None
        self.data_preparation = None

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
            self.data_preparation = DataPreparation(self.selected_folder)
            self.print_to_output(f'DataPreparation initialized with folder: {self.selected_folder}')

    def print_to_output(self, text):
        self.output_field.append(text)  # Fügt Text am Ende des QTextEdit hinzu

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = FolderSelector()
    ex.show()
    sys.exit(app.exec_())
