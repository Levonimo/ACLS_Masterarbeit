from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, 
                             QListWidget, QCheckBox, QGridLayout, QGroupBox)

from .groupmaker import GroupMaker
from .components import CheckableComboBox

from . import styles

# =========================================================================================================
# Input Dialog
# =========================================================================================================

class InputDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle('New File-Name')
        self.setGeometry(200, 200, 300, 100)

        # Layout
        self.setStyleSheet(styles.Levin)

        
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


# =========================================================================================================
# File Selection Window
# =========================================================================================================

class FileSelectionWindow(QDialog):
    def __init__(self, file_names, parent):
        super().__init__(parent)
        self.setWindowTitle('Select a File')
        self.setGeometry(150, 150, 400, 300)

        layout = QVBoxLayout()

        # Layout
        self.setStyleSheet(styles.Levin)

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

# =========================================================================================================
# Warping Selection Window
# =========================================================================================================
'''
To do:
    - Make a Windowfor File Selection with checkboxes
    - Add a checkbox for each file name in the list
'''

# =========================================================================================================
# Crossref File Selection Window
# =========================================================================================================

class CrossrefFileSelectionWindow(QDialog):
    def __init__(self, file_names, parent):
        super().__init__(parent)
        self.setWindowTitle('Select a File')
        self.setGeometry(150, 150, 400, 300)

        layout = QVBoxLayout()

        # Layout
        self.setStyleSheet(styles.Levin)
        
        fileGroupbox = QGroupBox('Select files', self)
        fileGroupboxLayout = QGridLayout(fileGroupbox) 
        self.checkbox_dict = {}
       
        for index, file_name in enumerate(file_names):
            checkbox = QCheckBox(file_name, self)
            row = index % 10
            col = index // 10
            fileGroupboxLayout.addWidget(checkbox, row, col)
            self.checkbox_dict[file_name] = checkbox

        # add a new column for checkboxes and dropdown menu with groupboxes

        # button to select all files at once
        self.select_all_button = QPushButton('Select all', self)
        fileGroupboxLayout.addWidget(self.select_all_button, 0, col+1, 1, 1)
        self.select_all_button.clicked.connect(self.select_all)

        # button to deselect all files at once
        self.deselect_all_button = QPushButton('Deselect all', self)
        fileGroupboxLayout.addWidget(self.deselect_all_button, 1, col+1, 1, 1)
        self.deselect_all_button.clicked.connect(self.deselect_all)

        # Dropdwoen menu for group selection with 
        Groups, filename_parts = GroupMaker(file_names)
        
        for i, key in enumerate(Groups.keys()):
            Groups[key] = sorted(Groups[key])
            group_dropdown = CheckableComboBox(self.checkbox_action(self, key, filename_parts), self)
            group_dropdown.addItems(Groups[key])
            fileGroupboxLayout.addWidget(group_dropdown, 2+i, col+1, 1, 1)

        
        self.select_button = QPushButton('Select')
        self.select_button.clicked.connect(self.select_file)
        fileGroupboxLayout.addWidget(self.select_button, 9, col+1, 1, 1)

        fileGroupbox.setLayout(fileGroupboxLayout)
        layout.addWidget(fileGroupbox)




    def select_file(self):
        selected_files = [file_name for file_name, checkbox in self.checkbox_dict.items() if checkbox.isChecked()]
        self.selected_files = selected_files
        self.accept()

    # Functions for the checkboxes
    def select_all(self):
        for checkbox in self.checkbox_dict.values():
            checkbox.setChecked(True)

    def deselect_all(self):
        for checkbox in self.checkbox_dict.values():
            checkbox.setChecked(False)
    
    class checkbox_action:
        def __init__(self, parent, group_index, filename_parts) -> None:
            self.parent = parent
            self.group_index = group_index
            self.filename_parts = filename_parts

        def trigger(self, item_text, checked):
            for file_name, parts in self.filename_parts.items():
                if parts[self.group_index] == item_text:
                    self.parent.checkbox_dict[file_name].setChecked(checked)