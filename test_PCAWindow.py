import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget
from component.GUI_Analysis import PCAWindow
import numpy as np
from component.fun_Groupmaker import GroupMaker

sys.path.insert(1, os.path.dirname(__file__))

class NormalGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.selected_folder = "U:/Documents/Masterarbeit/Daten_MA"
        # self.selected_folder = "F:/Dokumente/Masterarbeit/MA_Daten"

        warped = np.load('./output/warped_chromatograms.npy', allow_pickle=True).item()
        # print(warped)
        unwarped = np.load('./output/unwarped_chromatograms.npy', allow_pickle=True).item()
        # print(unwarped)
        targets = np.load('./output/selected_target.npy', allow_pickle=True)
        # print(targets)
        rt = np.load('./output/retention_time.npy', allow_pickle=True)

        self.Groups, self.filename_parts = GroupMaker(targets)

        # mz from range 
        mz_list = np.round(np.arange(35, 400.1, 1), 1)
        
        self.run_id = 1

        # self.selected_reference_file = '051_A2_5_SGL'
        self.selected_reference_file = '052_A2_6_SGL'            
        # Example usage of PCAWindow
        pca_window = PCAWindow(targets, warped, unwarped ,rt, mz_list, parent = self)
        if pca_window.exec_():
            print(f'PCA Results: {pca_window.results}')
        else:
            # stop everything if the window is closed
            print('PCA Window was closed without selection.')
        self.decline()

    def decline(self):
        self.reject()


def main():
    app = QApplication(sys.argv)

    wnd = NormalGUI()
    # if normal gui call self.close stop the program
    if not wnd.exec_():
        print('Normal GUI was closed without selection.')
        sys.exit(0)
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

