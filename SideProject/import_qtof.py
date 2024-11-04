from pyopenms import MSExperiment, MzMLFile
import h5py
import numpy as np
import os
import subprocess


# Funktion zum Laden der mzML-Datei und Speichern in HDF5
def mzml_to_hdf5(mzml_filename, hdf5_filename):
    # Erstelle ein MSExperiment-Objekt, um die mzML-Daten zu speichern
    exp = MSExperiment()
    mzml_file = MzMLFile()
    mzml_file.load(mzml_filename, exp)

    # Speichere die Daten in einer HDF5-Datei
    with h5py.File(hdf5_filename, "w") as f:
        # Durchlaufe die Spektren und speichere m/z- und Intensitätswerte
        for i, spectrum in enumerate(exp.getSpectra()):
            group = f.create_group(f"spectrum_{i}")
            mz_values, intensity_values = spectrum.get_peaks()
            group.create_dataset("mz", data=np.array(mz_values))
            group.create_dataset("intensity", data=np.array(intensity_values))
    print(f"Die Daten wurden erfolgreich in '{hdf5_filename}' gespeichert.")


def convert_d_to_mzml(path):
    mzml_path = os.path.join(path, "mzml/")
    for file in os.listdir(path):
        if file.endswith(".D"):
            mzml_file = os.path.join(mzml_path, file.replace(".D", ".mzML"))
            if not os.path.exists(mzml_file):
                command = f"msconvert {os.path.join(path, file)} -o {mzml_path} --mzML"
                subprocess.run(command, shell=True, check=True)


# Define path to the mzML file
PATH = 'F:/Documents/MasterArbeit/QTOF_DATA/'
EXPORT_PATH = 'F:/Documents/MasterArbeit/ACLS_Masterarbeit/SideProject/Data/'
convert_d_to_mzml(PATH)
FILENAME = 'Larve_washed_02.mzML'
# Beispiel: Verwende die Funktion
mzml_to_hdf5(PATH+"mzml/"+FILENAME, EXPORT_PATH+"daten.h5")
