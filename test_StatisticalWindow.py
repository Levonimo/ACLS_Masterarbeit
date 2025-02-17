import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QGridLayout
from component.GUI_Statistic import StatisticalWindow
import numpy as np

sys.path.insert(1, os.path.dirname(__file__))

class NormalGUI(QWidget):
        def __init__(self):
            super().__init__()
            self.selected_folder = "U:/Documents/Masterarbeit/Daten_MA"

            self.results = np.load(self.selected_folder + '/output/PCA_results.npy', allow_pickle=True).item()
            #print(self.results)
            self.file_names = self.results['scores'].keys()
            #unwarped = np.load('./Outputs/unwarped_chromatograms.npy', allow_pickle=True).item()
            
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
            pca_window = StatisticalWindow(self.results, self.file_names, parent = self)
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

