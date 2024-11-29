from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, 
                             QListWidget, QCheckBox, QComboBox,QGridLayout, QGroupBox,
                             QDesktopWidget)

from .PCA import perform_pca
from .styles_pyqtgraph import graph_style_chromatogram
import pyqtgraph as pg

# =========================================================================================================
# Input Dialog

class InputDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('New File-Name')
        self.setGeometry(200, 200, 300, 100)

        # Calculate the center position of the parent window
        if parent:
            parent_geometry = parent.geometry()
            parent_center_x = parent_geometry.x() + parent_geometry.width() // 2
            parent_center_y = parent_geometry.y() + parent_geometry.height() // 2
            
            # Get the screen geometry of the monitor where the parent window is located
            screen = QDesktopWidget().screenGeometry(parent)
            
            # Move the new window to the center of the parent window within the screen
            self.move(screen.x() + parent_center_x - self.width() // 2, 
                      screen.y() + parent_center_y - self.height() // 2)
        
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

# =========================================================================================================
# Warping Selection Window
'''
To do:
    - Make a Windowfor File Selection with checkboxes
    - Add a checkbox for each file name in the list
'''

# =========================================================================================================
# PCA Window

class PCAWindow(QDialog):
    def __init__(self, file_names: list, warped: dict, unwarped: dict, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle('PCA Settings')
        self.setMinimumSize(400, 400)
        # set popup place on top of the main window
        

        layout = QGridLayout()

        # Add checkbox for each file_name in a grouped layout
        fileGroupbox = QGroupBox('Select files', self)
        fileGroupboxLayout = QGridLayout(fileGroupbox)
        
        self.checkbox_dict = {}
        
        for index, file_name in enumerate(file_names):
            checkbox = QCheckBox(file_name, self)
            row = index % 10
            col = index // 10
            fileGroupboxLayout.addWidget(checkbox, row, col)
            self.checkbox_dict[file_name] = checkbox


        ''' 
        to do:  
            - add Group Selcetion for the files, for example all files with the same ending should be selected at once
            - add a button to select all files at once
            - add button to deselect all files at once

            - if possible add doubles checkbox 
                --> first all boxes are unchecked, click once on a box, the box is checked, click again on the box, the box is marked as double

        
        '''
        
        fileGroupbox.setLayout(fileGroupboxLayout)
        layout.addWidget(fileGroupbox, 0, 0, 1, 1)


        ParametersGroupbox = QGroupBox('Parameters', self)
        ParametersLayout = QVBoxLayout(ParametersGroupbox)

        # Add input fields for the number of components
        self.label = QLabel('Number of components:', self)
        ParametersLayout.addWidget(self.label)
        # input field for number of components with a default value of 5
        self.input_field = QLineEdit(self)
        self.input_field.setText('5')
        ParametersLayout.addWidget(self.input_field)

        # Add dropdown menu to choose the mehode
        self.label = QLabel('Method:', self)
        ParametersLayout.addWidget(self.label)

        self.method_dropdown = QComboBox(self)
        self.method_dropdown.addItems(['svd', 'eigen'])
        ParametersLayout.addWidget(self.method_dropdown)

        #Add dropdown menu to choose scaler method
        self.label = QLabel('Scaler:', self)
        ParametersLayout.addWidget(self.label)

        self.scaler_dropdown = QComboBox(self)
        self.scaler_dropdown.addItems(['None','StandardScaler', 'MinMaxScaler', 'MaxAbsScaler', 'RobustScaler'])
        ParametersLayout.addWidget(self.scaler_dropdown)

        # Add Dropdown menu to choose if Warped or Unwarped data should be used
        self.label = QLabel('Data from:', self)
        ParametersLayout.addWidget(self.label)

        self.data_dropdown = QComboBox(self)
        self.data_dropdown.addItems(['Warped', 'Unwarped'])
        ParametersLayout.addWidget(self.data_dropdown)

        # Add submit button
        self.submit_button = QPushButton('Performe PCA', self)
        ParametersLayout.addWidget(self.submit_button)
        self.submit_button.clicked.connect(self.submit)

        # Add close button
        self.close_button = QPushButton('Close', self)
        ParametersLayout.addWidget(self.close_button)
        self.close_button.clicked.connect(self.close)

        ParametersGroupbox.setLayout(ParametersLayout)
        layout.addWidget(ParametersGroupbox, 0, 1, 1, 1)


        self.setLayout(layout)
              

        # Add Groupbox for the plots
        PlotGroupBox = QGroupBox("Plots", self)
        PlotLayout = QGridLayout()

        # Plot Windows for the chromatograms with pyqtgraph
        self.plot_graph_left = pg.PlotWidget()
        graph_style_chromatogram(self.plot_graph_left)
        PlotLayout.addWidget(self.plot_graph_left, 0, 0, 1, 1)

        self.plot_graph_right = pg.PlotWidget()
        graph_style_chromatogram(self.plot_graph_right)
        PlotLayout.addWidget(self.plot_graph_right, 0, 1, 1, 1)

        
        PlotGroupBox.setLayout(PlotLayout)
        layout.addWidget(PlotGroupBox, 1, 0, 1, 1)

        # Add Groupbox for saving the results
        ResultGroupBox = QGroupBox("Results", self)
        ResultLayout = QVBoxLayout(ResultGroupBox)

        self.result_label = QLabel('Path to results', self)
        ResultLayout.addWidget(self.result_label)

        self.save_button = QPushButton('Save', self)
        ResultLayout.addWidget(self.save_button)

        ResultGroupBox.setLayout(ResultLayout)
        layout.addWidget(ResultGroupBox, 1, 1, 1, 1)



        self.input_text = None
        self.selected_files = None
        self.method = None
        self.scaler = None
        self.data = None
        self.warped_data = warped
        self.unwarped_data = unwarped   
        self.results = None

        # Colors by file name endings
        self.colors = {
            'SOO': (255, 0, 0, 255), # light red
            'SOL': (139, 0, 0, 255), # dark red
            'SGO': (0, 255, 0, 255), # light green,
            'SGL': (0, 139, 0, 255), # dark green,
            'OOO': (0, 0, 255, 255), # light blue
            'FFF': (0, 0, 139, 255) # dark blue
        }
        

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


        # Perform PCA with the given parameters, only the selected files
        # should be used for the PCA
        scores, loadings, explained_variance = perform_pca(selected_data, n_components, self.scaler, self.method)

        # The results should be stored in the results attribute
        self.results = {
            'scores': scores,
            'loadings': loadings,
            'explained_variance': explained_variance
        }
        
        # The result should be displayed in the result placeholder
        # The result should be a scatter plot of the first two components
        # of the PCA, colored by the file name endings 
        self.plot_graph_left.clear()
        
        for i, sample in enumerate(self.selected_files):
            color = self.colors[sample[-3:]]
            self.plot_graph_left.plot([scores[i, 0]], [scores[i, 1]], pen=None, symbol='o', symbolBrush=color)

        self.plot_graph_left.setLabel('bottom', 'Component 1')
        self.plot_graph_left.setLabel('left', 'Component 2')

        # Plot the explained variance as Barplot
        self.plot_graph_right.clear()
        bar_graph = pg.BarGraphItem(x=range(1, n_components+1), height=explained_variance, width=0.6, brush='b')
        self.plot_graph_right.addItem(bar_graph)
        # add total explained variance as line plot 
        self.plot_graph_right.plot(range(1, n_components+1), explained_variance.cumsum(), pen=pg.mkPen(color=(255, 0, 0)))
        self.plot_graph_right.setLabel('left', 'Explained Variance')
        self.plot_graph_right.setLabel('bottom', 'Component')

            



       






        