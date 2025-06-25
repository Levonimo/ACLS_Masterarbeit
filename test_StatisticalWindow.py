import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QGridLayout
from component.GUI_Statistic import StatisticalWindow
import numpy as np
from component import styles

sys.path.insert(1, os.path.dirname(__file__))

class NormalGUI(QWidget):
    def __init__(self):
        super().__init__()
        # self.selected_folder = "F:/Dokumente/Masterarbeit/MA_Daten/output/"
        self.selected_folder = "U:/Documents/Masterarbeit/250521_Larva_SPME/output/"

        # self.results = np.load(self.selected_folder + 'PCA_results_7b9bc4f6-8904-4dc8-a13d-e06469a6f4e1.npy', allow_pickle=True).item()
        self.results = np.load(self.selected_folder + 'PCA_results_08b4807f-9c12-4d63-9e86-ad5974e26f5c.npy', allow_pickle=True).item()
        #print(self.results)

        # use styles from styles.py on current window
        self.setStyleSheet(styles.Levin)


        self.setWindowTitle('Statistical Analysis')
    

        layout = QGridLayout()
        self.setLayout(layout)
        # add  a botton to the window
        self.button = QPushButton('Open Statistical Window', self)
        self.button.clicked.connect(self.open_statistical_window)
        layout.addWidget(self.button, 0, 0)


        
        self.run_id = 1
        

    def open_statistical_window(self):
        # Example usage of PCAWindow
        pca_window = StatisticalWindow(self.results, parent = self)
        if pca_window.exec_():
            print(f'PCA Results: {pca_window.results}')
            pass


            


def main():
    app = QApplication(sys.argv)

    wnd = NormalGUI()
    wnd.show()
    

    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

