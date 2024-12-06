from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, 
                             QListWidget, QCheckBox, QComboBox,QGridLayout, QGroupBox,
                             QDesktopWidget, QListWidgetItem)

from PyQt5.QtCore import Qt

from .PCA import perform_pca
from .styles_pyqtgraph import graph_style_chromatogram
import pyqtgraph as pg
from .groupmaker import GroupMaker
from .components import CheckableComboBox
import numpy as np

# =========================================================================================================
# Input Dialog
# =========================================================================================================

class InputDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle('New File-Name')
        self.setGeometry(200, 200, 300, 100)

        # Calculate the center position of the parent window
        
        
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
        layout.addWidget(fileGroupbox, 0, 0, 1, 1)




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




# =========================================================================================================
# PCA Window
# =========================================================================================================

class PCAWindow(QDialog):
    def __init__(self, file_names: list, warped: dict, unwarped: dict, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle('PCA Settings')
        self.setMinimumSize(400, 400)
        

        layout = QGridLayout()

        #=========================================================================================================
        ### Add Groupbox for the files
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




        ''' 
        to do:
         - if possible add doubles checkbox 
                --> first all boxes are unchecked, click once on a box, the box is checked, click again on the box, the box is marked as double

        
        '''
        
        fileGroupbox.setLayout(fileGroupboxLayout)
        layout.addWidget(fileGroupbox, 0, 0, 1, 1)

        #=========================================================================================================
        # Add Groupbox for the parameters
        ParametersGroupbox = QGroupBox('Parameters', self)
        ParametersLayout = QGridLayout(ParametersGroupbox)

        # add dropdown menu to select 2D chromatogrmm or 3D chromatogram
        self.label = QLabel('Chromatogram Dimension:', self)
        ParametersLayout.addWidget(self.label, 0, 0, 1, 1)

        self.chrom_dropdown = QComboBox(self)
        self.chrom_dropdown.addItems(['2D', '3D'])
        ParametersLayout.addWidget(self.chrom_dropdown, 0, 1, 1, 1)

        # Add input fields for the number of components
        self.label = QLabel('Number of components:', self)
        ParametersLayout.addWidget(self.label, 1, 0, 1, 1)
        # input field for number of components with a default value of 5
        self.input_number_PC = QLineEdit(self)
        self.input_number_PC.setText('5')
        ParametersLayout.addWidget(self.input_number_PC, 1, 1, 1, 1)

        # Add dropdown menu to choose the mehode
        self.label = QLabel('Method:', self)
        ParametersLayout.addWidget(self.label, 2, 0, 1, 1)

        self.method_dropdown = QComboBox(self)
        self.method_dropdown.addItems(['svd', 'eigen'])
        ParametersLayout.addWidget(self.method_dropdown, 2, 1, 1, 1)

        #Add dropdown menu to choose scaler method
        self.label = QLabel('Scaler:', self)
        ParametersLayout.addWidget(self.label, 3, 0, 1, 1)

        self.scaler_dropdown = QComboBox(self)
        self.scaler_dropdown.addItems(['None','StandardScaler', 'MinMaxScaler', 'MaxAbsScaler', 'RobustScaler'])
        ParametersLayout.addWidget(self.scaler_dropdown, 3, 1, 1, 1)

        # Add Dropdown menu to choose if Warped or Unwarped data should be used
        self.label = QLabel('Data from:', self)
        ParametersLayout.addWidget(self.label, 4, 0, 1, 1)

        self.data_dropdown = QComboBox(self)
        self.data_dropdown.addItems(['Warped', 'Unwarped'])
        ParametersLayout.addWidget(self.data_dropdown, 4, 1, 1, 1)

        # Add submit button
        self.submit_button = QPushButton('Performe PCA', self)
        ParametersLayout.addWidget(self.submit_button, 5, 1, 1, 1)
        self.submit_button.clicked.connect(self.submit)

        # Add close button
        self.close_button = QPushButton('Close', self)
        ParametersLayout.addWidget(self.close_button, 5, 0, 1, 1)
        self.close_button.clicked.connect(self.close)
        
        # add checkboxes for group selection






        ParametersGroupbox.setLayout(ParametersLayout)
        layout.addWidget(ParametersGroupbox, 0, 1, 1, 1)

        #=========================================================================================================
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

        #=========================================================================================================
        # Add Groupbox for saving and displaying the results
        ResultGroupBox = QGroupBox("Results", self)
        ResultLayout = QGridLayout(ResultGroupBox)

        self.number_PC = self.input_number_PC.text()

        # add elements for score plot
        self.label = QLabel('Component for X-Axis:', self)
        ResultLayout.addWidget(self.label, 0, 0, 1, 1)
        self.score_xaxis_dropdown = QComboBox(self)
        self.score_xaxis_dropdown.addItems([f'Component {i+1}' for i in range(int(self.number_PC))])
        ResultLayout.addWidget(self.score_xaxis_dropdown, 0, 1, 1, 1)

        self.label = QLabel('Component for Y-Axis:', self)
        ResultLayout.addWidget(self.label, 1, 0, 1, 1)
        self.score_yaxis_dropdown = QComboBox(self)
        self.score_yaxis_dropdown.addItems([f'Component {i+1}' for i in range(int(self.number_PC))])
        ResultLayout.addWidget(self.score_yaxis_dropdown, 1, 1, 1, 1)

        self.score_plot_button = QPushButton('Display Scores', self)
        ResultLayout.addWidget(self.score_plot_button, 2, 1, 1, 1)
        self.score_plot_button.clicked.connect(self.display_scores)


        # add elements for loadings plot
        self.loadings_compound_dropdown = QComboBox(self)
        # add item for each principal component
        self.loadings_compound_dropdown.addItems([f'Component {i+1}' for i in range(int(self.number_PC))])
        ResultLayout.addWidget(self.loadings_compound_dropdown, 4, 1, 1, 1)

        # add a button to display the loadings of the selected component
        self.loadings_button = QPushButton('Display Loadings', self)
        ResultLayout.addWidget(self.loadings_button, 5, 1, 1, 1)
        self.loadings_button.clicked.connect(self.display_loadings)


        #### Test selection of files for crossref
        self.crossref_button = QPushButton('Crossref Files', self)
        ResultLayout.addWidget(self.crossref_button, 6, 1, 1, 1)
        self.crossref_button.clicked.connect(self.crossref_files)



        # add elements for Saving the results
        self.result_label = QLabel('Path to results', self)
        ResultLayout.addWidget(self.result_label)

        self.save_button = QPushButton('Save', self)
        ResultLayout.addWidget(self.save_button)

        ResultGroupBox.setLayout(ResultLayout)
        layout.addWidget(ResultGroupBox, 1, 1, 1, 1)

        #=========================================================================================================
        # Add Groupbox for Loadings plot
        LoadingsGroupBox = QGroupBox("Loadings", self)
        LoadingsLayout = QGridLayout(LoadingsGroupBox)

        self.loadings_plot = pg.PlotWidget()
        graph_style_chromatogram(self.loadings_plot)
        LoadingsLayout.addWidget(self.loadings_plot)

        

        LoadingsGroupBox.setLayout(LoadingsLayout)
        layout.addWidget(LoadingsGroupBox, 2, 0, 1, 2)


        self.setLayout(layout)


        #=========================================================================================================
        # initialize the attributes
        
        self.number_PC = None
        self.selected_files = None
        self.method = None
        self.scaler = None
        self.data = None
        self.warped_data = warped
        self.unwarped_data = unwarped   
        self.results = None

        #=========================================================================================================
        # Colors by file name endings
        self.colors = {
            'SOO': (255, 0, 0, 255), # light red
            'SOL': (139, 0, 0, 255), # dark red
            'SGO': (0, 255, 0, 255), # light green,
            'SGL': (0, 139, 0, 255), # dark green,
            'OOO': (0, 0, 255, 255), # light blue
            'FFF': (0, 0, 139, 255) # dark blue
        }
        
    #=========================================================================================================
    #=========================================================================================================
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
         
        

    #=========================================================================================================
    #=========================================================================================================
    # Functions for the buttons
    def close(self):

        self.accept()


    def submit(self):
        self.number_PC = self.input_number_PC.text()
        self.selected_files = [file_name for file_name, checkbox in self.checkbox_dict.items() if checkbox.isChecked()]
        self.method = self.method_dropdown.currentText()
        self.scaler = self.scaler_dropdown.currentText()
        self.data_from = self.data_dropdown.currentText()
        self.chrom_dim = self.chrom_dropdown.currentText()
        n_components = int(self.number_PC)
        
        # print("warped data:", self.warped_data)

        # print("Datatyoe of warped data:", type(self.warped_data))


        # Load Warped or Unwarped data depending on the selected filenames
        if self.data_from == 'Warped':  
            data = self.warped_data
        else:
            data = self.unwarped_data
            
        # print("Data:", data)
        # print("type of data:", type(data))

        # print("Selected Files:", self.selected_files)
        # print("Data Keys:", data.keys())

        selected_data = {key: data[key] for key in self.selected_files if key in data}
        # print("Selected Data:", selected_data)


        # Perform PCA with the given parameters, only the selected files
        # should be used for the PCA
        scores, loadings, explained_variance = perform_pca(selected_data, n_components, self.scaler, self.method, self.chrom_dim)


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
        self.plot_graph_left.enableAutoRange()        
        for i, sample in enumerate(self.selected_files):
            color = self.colors[sample[-3:]]
            self.plot_graph_left.plot([scores[i, 0]], [scores[i, 1]], pen=None, symbol='o', symbolBrush=color)

        self.plot_graph_left.setLabel('bottom', 'Component 1')
        self.plot_graph_left.setLabel('left', 'Component 2')

         # get all unique file endings
        unique_endings = list(set([file_name[-3:] for file_name in self.selected_files]))
        # add legend of all unique file endings
        legend = self.plot_graph_left.addLegend()
        # clear legend
        legend.clear()
        legend.setOffset((10, 10))  # Adjust the position of the legend if needed

        for ending in unique_endings:
            legend.addItem(pg.ScatterPlotItem(pen=None, brush=self.colors[ending], size=10), f'{ending}')

        # Plot the explained variance as Barplot
        self.plot_graph_right.clear()
        bar_graph = pg.BarGraphItem(x=range(1, n_components+1), height=explained_variance, width=0.6, brush='b')
        self.plot_graph_right.addItem(bar_graph)
        # add total explained variance as line plot 
        self.plot_graph_right.plot(range(1, n_components+1), explained_variance.cumsum(), pen=pg.mkPen(color=(255, 0, 0)))
        self.plot_graph_right.setLabel('left', 'Explained Variance')
        self.plot_graph_right.setLabel('bottom', 'Component')
        
       
              



        # Update checkbox for number of PC in loadings plot
        self.loadings_compound_dropdown.clear()
        self.loadings_compound_dropdown.addItems([f'Component {i+1}' for i in range(n_components)])
        self.score_xaxis_dropdown.clear()
        self.score_xaxis_dropdown.addItems([f'Component {i+1}' for i in range(n_components)])
        self.score_yaxis_dropdown.clear()
        self.score_yaxis_dropdown.addItems([f'Component {i+1}' for i in range(n_components)])


    def display_loadings(self):
        # Get the selected component
        component = self.loadings_compound_dropdown.currentIndex()
        rt=np.load('./Outputs/retention_time.npy', allow_pickle=True)
        # Get the loadings of the selected component
        if self.chrom_dim == '2D':
            loadings = self.results['loadings'][component]
        elif self.chrom_dim == '3D':
            loadings = np.sum(self.results['loadings'][component], axis=1)
        
        # Plot the loadings
        self.loadings_plot.clear()
        # reset zoom
        self.loadings_plot.enableAutoRange()
        self.loadings_plot.plot(rt, loadings, pen=pg.mkPen(color=(0, 0, 0)))
        self.loadings_plot.setLabel('bottom', 'Retention Time')
        self.loadings_plot.setLabel('left', 'Loading')

    def display_scores(self):
        # Get the selected components
        xaxis = self.score_xaxis_dropdown.currentIndex()
        yaxis = self.score_yaxis_dropdown.currentIndex()
        # Get the scores
        scores = self.results['scores']
        # Plot the scores
        self.plot_graph_left.clear()
        self.loadings_plot.enableAutoRange()

        for i, sample in enumerate(self.selected_files):
            color = self.colors[sample[-3:]]
            self.plot_graph_left.plot([scores[i, xaxis]], [scores[i, yaxis]], pen=None, symbol='o', symbolBrush=color)
        
        self.plot_graph_left.setLabel('bottom', f'Component {xaxis+1}')
        self.plot_graph_left.setLabel('left', f'Component {yaxis+1}')

        # get all unique file endings
        unique_endings = list(set([file_name[-3:] for file_name in self.selected_files]))
        # add legend of all unique file endings
        legend = self.plot_graph_left.addLegend()
        legend.clear()
        legend.setOffset((10, 10))  # Adjust the position of the legend if needed

        for ending in unique_endings:
            legend.addItem(pg.ScatterPlotItem(pen=None, brush=self.colors[ending], size=10), f'{ending}')


    def crossref_files(self):
        # get all unchecked files
        unchecked_files = [file_name for file_name, checkbox in self.checkbox_dict.items() if not checkbox.isChecked()]

        crossref_window = CrossrefFileSelectionWindow(unchecked_files, self)
        if crossref_window.exec_():
            crossref_files = crossref_window.selected_files

            # calculate score for the selected files with the current loadings
            # add them to the score plot with label 'ref'
            for file in crossref_files:
                color = self.colors[file[-3:]]
                score = np.dot(self.results['loadings'], self.results['scores'][self.selected_files.index(file)])
                self.plot_graph_left.plot([score[self.score_xaxis_dropdown.currentIndex()]], [score[self.score_yaxis_dropdown.currentIndex()]], pen=None, symbol='o', symbolBrush=color, symbolPen=None, symbolSize=10, name='ref')
            
