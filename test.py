import sys
import os
from PyQt5.QtWidgets import QApplication
from component.ExternalGui import PCAWindow
import numpy as np

sys.path.insert(1, os.path.dirname(__file__))

def main():
    app = QApplication(sys.argv)
    
    warped = np.load('./Outputs/warped_chromatograms.npy', allow_pickle=True).item()
    # print(warped)
    unwarped = np.load('./Outputs/unwarped_chromatograms.npy', allow_pickle=True).item()
    # print(unwarped)
    targets = np.load('./Outputs/selected_target.npy', allow_pickle=True)
    # print(targets)
    
    
    # Example usage of PCAWindow
    pca_window = PCAWindow(targets, warped, unwarped)
    if pca_window.exec_():
        print(f'PCA Results: {pca_window.results}')

    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

