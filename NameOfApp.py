"""
Name and description of the application

    In this section, you can provide a brief description of the application and its purpose.

    In this show who wrote and title of master thesis, and the supervisor of the project.

NAmeOfApp.py (equivalent with main.py)
"""


__author__      = "Levin Willi"
__copyright__   = "Copyright 2025, Zurich University of Applied Sciences, Center for Analytics"
__credits__     = ["Susanne Kern", "Olivier Merlo", "Nicolas Imstepf"]
__license__     = "GPL3, images: CC BY-NC-SA 4.0"
__version__     = "1.0.0"
__maintainer__  = "Levin Willi"
__email__       = "levinwilli@protonmail.ch"


import os
import sys

# add betplip path to directory 
sys.path.insert(1, os.path.dirname(__file__))

from component import MainWindow


# global reference to avoid garbage collection of our dialog
dialog = None


# show application window by running betaplip.py (developing purpose)
if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    print("MAIN")
    app = QApplication(sys.argv)
    wnd = MainWindow()
    wnd.show()
    sys.exit(app.exec_())
