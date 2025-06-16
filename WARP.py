"""
# WARP.py
# WARP - Warping Algorithm for Retention Profiles to Optimize TD-CIS-GC-MS Data for Statistical Analysis 
#
# This script is part of the WARP project, which is designed to optimize retention profiles for TD-CIS-GC-MS data.

"""


__author__      = "Levin Willi"
__copyright__   = "Copyright 2025, Zurich University of Applied Sciences, Center for Analytics, Material and Diagnostics"
__credits__     = ["Susanne Kern", "Olivier Merlo", "Nicolas Imstepf"]
__license__     = "GPL3: CC BY-NC-SA 4.0"
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
        subprocess.run("msconvert.exe", check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("MSConvert not found in PATH")
        return False
    return True

# Set openms_data_path and base_path
if getattr(sys, "frozen", False):
    try:
        base_path = __compiled_dir__ # type: ignore
    except NameError:
        base_path = os.path.dirname(sys.executable)
else:
    base_path = os.path.dirname(os.path.abspath(__file__))

# if 'OPENMS_DATA_PATH' not in os.environ:
#     potential_path = os.path.join(base_path, 'openms_data')
#     if os.path.isdir(potential_path):
#         os.environ['OPENMS_DATA_PATH'] = potential_path

# global reference to avoid garbage collection of our dialog
dialog = None

if __name__ == "__main__":
    # Check if MSConvert is in the PATH
    if not check_msconvert():
        print("MSConvert not found in PATH. Please install ProteoWizard and add Folder to PATH.")
        sys.exit(1)
    app = QApplication(sys.argv)
    # Pass base_path to your MainWindow constructor.
    window = MainWindow(base_path)
    window.show()
    sys.exit(app.exec_())