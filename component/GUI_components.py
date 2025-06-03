"""
Components of the Graphical user interface
    (Personalized PyQt objects)

MyBar is adapted from David Patsch's AlphaDock and Nicola Imstepf's Master Thesis.

components.py
"""

from . import styles, config

import webbrowser

from PyQt5.QtWidgets import (QHBoxLayout, QVBoxLayout, QPushButton, QWidget, QLayout, QLabel,
                             QMenu, QAction,  QComboBox, QTableWidget, QTableWidgetItem,QGridLayout,QDialog)
from PyQt5.QtCore import Qt, QPoint, QEvent
from PyQt5.QtGui import QStandardItemModel

import pyms.Display as Display
import matplotlib.pyplot as plt
#=========================================================================================================

class CheckableComboBox(QComboBox): 
    def __init__(self, action, parent=None): 
        super(CheckableComboBox, self).__init__(parent) 
        self.view().pressed.connect(self.handleItemPressed) 
        self.setModel(QStandardItemModel(self)) 
        self.action = action

    def addItems(self, items):
        for item in items:
            self.addItem(item)
            item = self.model().item(self.count() - 1)
            item.setCheckState(Qt.Unchecked)

    def handleItemPressed(self, index): 
        item = self.model().itemFromIndex(index) 
        if item.checkState() == Qt.Checked: 
            item.setCheckState(Qt.Unchecked) 
        else: 
            item.setCheckState(Qt.Checked)
        self.action.trigger(item.text(), item.checkState() == Qt.Checked)
       
#=========================================================================================================       
class MyBar(QWidget):
    """QWidget to replace the removed title bar and buttons"""

    def __init__(self, parent: QLayout) -> None:
        super(MyBar, self).__init__()
        
        # initialize general variables
        self.setStyleSheet(styles.Levin)      
        self.parent = parent
        
        # general settings
        layout = QHBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        btn_size = 40

        # left hand side
        leftWidget = QWidget()
        leftLayout = QHBoxLayout()
        leftLayout.setContentsMargins(0,0,0,0)

        # logo
        analyticlogo = QPushButton()
        analyticlogo.setStyleSheet(styles.analyticlogo)
        analyticlogo.setFixedSize(btn_size,btn_size)
        

        # settings
        btn_settings = QPushButton()
        btn_settings.setToolTip('Settings (Ctrl+I)')
        btn_settings.setShortcut("Ctrl+I")
        btn_settings.setFixedSize(btn_size,btn_size)
        btn_settings.setStyleSheet(styles.SettingsButton)

        self.color_choose = QAction("Choose color schema", self)
        self.color_choose.setShortcut("Ctrl+D")
        # click event open ColorChoosWindow
        self.color_choose.triggered.connect(lambda: ColorChoosWindow().show())
        

        self.actionclean_up = QAction("reset everything", self)
        self.actionclean_up.setShortcut("Ctrl+R")

        menu_setting = QMenu()
        menu_setting.setToolTipsVisible(True)
        menu_setting.setStyleSheet(styles.Levin)
        menu_setting.addAction(self.actionclean_up)
        menu_setting.addAction(self.color_choose)
        btn_settings.setMenu(menu_setting)

        # help
        btn_help = QPushButton()
        btn_help.clicked.connect(lambda: webbrowser.open(config.HELP_PATH))
        btn_help.setToolTip('Help (Ctrl+H)')
        btn_help.setFixedSize(btn_size, btn_size)
        btn_help.setStyleSheet(styles.MapButton)
        btn_help.setShortcut("Ctrl+H")

        leftLayout.addWidget(analyticlogo)
        leftLayout.addWidget(btn_settings)
        leftLayout.addWidget(btn_help)
        leftWidget.setLayout(leftLayout)
        layout.addWidget(leftWidget, alignment=Qt.AlignLeft)

        # right hand side
        rightWidget = QWidget()
        rightLayout = QHBoxLayout()
        rightLayout.setContentsMargins(0,0,0,0)

        # minimize window
        btn_min = QPushButton()
        btn_min.clicked.connect(lambda: self.parent.showMinimized())
        btn_min.setFixedSize(btn_size, btn_size)
        btn_min.setStyleSheet(styles.MinButton)

        # close window
        btn_close = QPushButton()
        btn_close.clicked.connect(lambda: self.parent.close())
        btn_close.setFixedSize(btn_size,btn_size)
        btn_close.setStyleSheet(styles.CloseButton)

        rightLayout.addWidget(btn_min)
        rightLayout.addWidget(btn_close)
        rightWidget.setLayout(rightLayout)
        layout.addWidget(rightWidget, alignment=Qt.AlignRight)

        self.setLayout(layout)

        # ability to move the window
        # adapted from from stackoverflow.com/questions/44241612
        self.start = QPoint(0, 0)
        self.pressing = False

    def mousePressEvent(self, event: QEvent) -> None:
        """
        ability to move the window
        adapted from from stackoverflow.com/questions/44241612
        """
        self.start = self.mapToGlobal(event.pos())
        self.pressing = True

    def mouseMoveEvent(self, event: QEvent) -> None:
        """
        ability to move the window
        adapted from from stackoverflow.com/questions/44241612
        """
        if self.pressing:
            self.end = self.mapToGlobal(event.pos())
            self.movement = self.end-self.start
            self.parent.setGeometry(self.mapToGlobal(self.movement).x()-10,
                                self.mapToGlobal(self.movement).y()-10,
                                self.parent.width(),
                                self.parent.height())
            self.start = self.end

# =========================================================================================================
# ColorChoosWindow
# =========================================================================================================


class ColorChoosWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Variable", "Hex Color", "Preview"])
        self.table.horizontalHeader().setStretchLastSection(True)

        # Initially insert one empty row
        self.insertRowWithPreview()

        # Connect itemChanged
        self.table.itemChanged.connect(self.handleChange)

        # Save Button
        button = QPushButton("Save")
        button.clicked.connect(self.saveColors)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.table)
        layout.addWidget(button)
        self.setLayout(layout)

    def insertRowWithPreview(self):
        """Inserts a new row and sets a QLabel in the Preview column."""
        row = self.table.rowCount()
        self.table.insertRow(row)

        # Create empty cells for "Variable" and "Hex Color"
        self.table.setItem(row, 0, QTableWidgetItem(""))
        self.table.setItem(row, 1, QTableWidgetItem(""))

        # Create a QLabel for the preview column
        label = QLabel()
        label.setAlignment(Qt.AlignCenter)
        self.table.setCellWidget(row, 2, label)

    def handleChange(self, item):
        """Update preview if the user changed the 'Hex Color' column, then ensure a new blank row."""
        if item.column() == 1:
            # Validate or update color
            self.updatePreview(item.row())

            # If both columns are filled in this row, add a new row
            self.ensureNewRow(item.row())

    def updatePreview(self, row):
        """Set the background color of the preview label to the user-entered text."""
        color_item = self.table.item(row, 1)
        preview_label = self.table.cellWidget(row, 2)
        if color_item and preview_label:
            color_text = color_item.text()
            # Basic validation for a hex color like "#RRGGBB"
            if len(color_text) == 7 and color_text.startswith("#"):
                preview_label.setStyleSheet(f"background-color: {color_text};")
            else:
                preview_label.setStyleSheet("")  # clear if invalid

    def ensureNewRow(self, row):
        """If both Variable and Hex Color columns in the current row are non-empty, insert another row."""
        v_item = self.table.item(row, 0)
        c_item = self.table.item(row, 1)
        if v_item and c_item and v_item.text() and c_item.text():
            # Insert a new row only if we don't already have a totally blank row at the bottom
            # (Alternatively, just always insert a row.)
            last_row = self.table.rowCount() - 1
            # Check if the last row is already blank
            if (self.table.item(last_row, 0) and not self.table.item(last_row, 0).text()) or \
               (self.table.item(last_row, 1) and not self.table.item(last_row, 1).text()):
                # There's already a blank row
                return
            self.insertRowWithPreview()

    def saveColors(self):
        """Collect all non-empty variable-color pairs and do something with them."""
        self.colors = {}
        for r in range(self.table.rowCount()):
            var_item = self.table.item(r, 0)
            col_item = self.table.item(r, 1)
            if var_item and col_item and var_item.text() and col_item.text():
                self.colors[var_item.text()] = col_item.text()
        print(self.colors)  # or do something more sophisticated
        self.close()



# =========================================================================================================
# DisplayMassplotWindow

class MassplotWindow(QDialog):
    def __init__(self, parent=None, name =None,Mass_Spec = None, Mass_Spec_Ref = None):
        super().__init__(parent)
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure
        self.setWindowTitle("Mass Spectra Comparison")
        self.setGeometry(100, 100, 800, 600)

        # Create a layout with grid
        layout = QGridLayout()
        self.setLayout(layout)
        # Create a canvas for the mass spectra
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas, 0, 0, 2, 2)
        # Create a button for Decline or Accept the comparison
        self.decline_button = QPushButton("Decline")
        self.decline_button.clicked.connect(self.decline)
        self.accept_button = QPushButton("Accept")  
        self.accept_button.clicked.connect(self.take)
        layout.addWidget(self.decline_button, 1, 0)     
        layout.addWidget(self.accept_button, 1, 1)
        
        self.flag = False

        # Plot the mass spectra
        self.figure.clf()
        ax = self.figure.add_subplot(111)
        ax.set_title(f"Mass Spectra Comparison: {name}")
        ax.set_xlabel("m/z")
        ax.set_ylabel("Intensity")
        Display.plot_head2tail(ax, Mass_Spec, Mass_Spec_Ref)
        self.figure.subplots_adjust(top=0.94, bottom=0.2, left=0.05, right=0.99)
        self.canvas.draw()



    def take(self):
        """Handle the accept button click."""
        self.flag = True
        self.accept()  # Call accept to set the flag

    def decline(self):
        """Handle the close button click."""
        self.reject()  # Close dialog with rejection