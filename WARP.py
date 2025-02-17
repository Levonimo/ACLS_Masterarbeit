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
import subprocess
from PyQt5.QtWidgets import QApplication
from component import MainWindow

# Function to check if MSConvert.exe is in the PATH
def check_msconvert():
    try:
        subprocess.run("msconvert", check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("MSConvert not found in PATH")
        return False
    return True

# Set openms_data_path and base_path
if getattr(sys, "frozen", False):
    try:
        base_path = __compiled_dir__
    except NameError:
        base_path = os.path.dirname(sys.executable)
else:
    base_path = os.path.dirname(os.path.abspath(__file__))

if 'OPENMS_DATA_PATH' not in os.environ:
    potential_path = os.path.join(base_path, 'openms_data')
    if os.path.isdir(potential_path):
        os.environ['OPENMS_DATA_PATH'] = potential_path

# global reference to avoid garbage collection of our dialog
dialog = None

if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Pass base_path to your MainWindow constructor.
    window = MainWindow(base_path)
    window.show()
    sys.exit(app.exec_())