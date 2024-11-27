from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, QListWidget, QCheckBox, QComboBox

from .PCA import perform_pca


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


class PCAWindow(QDialog):
    def __init__(self, file_names, warped, unwarped, parent=None):
        super().__init__(parent)
        self.setWindowTitle('PCA Settings')
        self.setGeometry(200, 200, 300, 100)

        layout = QVBoxLayout()

        # Add input fields for the number of components
        self.label = QLabel('Number of components:', self)
        layout.addWidget(self.label)

        self.input_field = QLineEdit(self)
        layout.addWidget(self.input_field)

        # Add checkbox for each file_name
        self.checkbox_dict = {}
        for file_name in file_names:
            checkbox = QCheckBox(file_name, self)
            layout.addWidget(checkbox)
            self.checkbox_dict[file_name] = checkbox

        # Add dropdown menu to choose the mehode
        self.label = QLabel('Method:', self)
        layout.addWidget(self.label)

        self.method_dropdown = QComboBox(self)
        self.method_dropdown.addItems(['svd', 'eigen'])
        layout.addWidget(self.method_dropdown)

        #Add dropdown menu to choose scaler method
        self.label = QLabel('Scaler:', self)
        layout.addWidget(self.label)

        self.scaler_dropdown = QComboBox(self)
        self.scaler_dropdown.addItems(['StandardScaler', 'MinMaxScaler', 'MaxAbsScaler', 'RobustScaler'])
        layout.addWidget(self.scaler_dropdown)

        # Add Dropdown menu to choose if Warped or Unwarped data should be used
        self.label = QLabel('Data from:', self)
        layout.addWidget(self.label)

        self.data_dropdown = QComboBox(self)
        self.data_dropdown.addItems(['Warped', 'Unwarped'])
        layout.addWidget(self.data_dropdown)

        # Add submit button
        self.submit_button = QPushButton('Performe PCA', self)
        layout.addWidget(self.submit_button)
        self.submit_button.clicked.connect(self.submit)

        self.setLayout(layout)
        self.input_text = None
        self.selected_files = None
        self.method = None
        self.scaler = None
        self.data = None

        # Add close button
        self.close_button = QPushButton('Close', self)
        layout.addWidget(self.close_button)
        self.close_button.clicked.connect(self.close)

        # Add placeholder for the results as image, two plots next to each other
        self.result = QLabel(self)
        layout.addWidget(self.result)

        self.warped_data = warped
        self.unwarped_data = unwarped       

    def close(self):
        self.accept()


    def submit(self):
        self.input_text = self.input_field.text()
        self.selected_files = [file_name for file_name, checkbox in self.checkbox_dict.items() if checkbox.isChecked()]
        self.method = self.method_dropdown.currentText()
        self.scaler = self.scaler_dropdown.currentText()
        self.data_from = self.data_dropdown.currentText()

        # Load Warped or Unwarped data depending on the selectet filenames
        if self.data_from == 'Warped':  
            data = self.warped_data
            selected_data = {key: data[key] for key in self.selected_files if key in data}
        else:
            data = self.unwarped_data
            selected_data = {key: data[key] for key in self.selected_files if key in data}

        n_components = int(self.input_text)
        result = perform_pca(selected_data, n_components, self.scaler, self.method)

        # Perform PCA with the given parameters, only the selected files
        # should be used for the PCA




        # Perform PCA with the given parameters, only the selected files
        # should be used for the PCA
        # The result should be displayed in the result placeholder
        # The result should be a scatter plot of the first two components
        # of the PCA, colored by the file name and a bar plot of the explained
        # variance ratio of the components






        