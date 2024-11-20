from pyopenms import MSExperiment, MzMLFile

import h5py
import numpy as np
import os
import subprocess
import dask.dataframe as dd
import time as t
import pandas as pd

# Funktion zum Laden der mzML-Datei und Speichern in HDF5
def mzml_to_hdf5(mzml_filename, hdf5_filename):
    # Erstelle ein MSExperiment-Objekt, um die mzML-Daten zu speichern
    exp = MSExperiment()
    mzml_file = MzMLFile()
    mzml_file.load(mzml_filename, exp)

    # Speichere die Daten in einer HDF5-Datei
    with h5py.File(hdf5_filename, "w") as f:
        # Durchlaufe die Spektren und speichere m/z- und Intensitätswerte 
        for spectrum in exp.getSpectra():
            rt = spectrum.getRT()
            group = f.create_group(str(rt))
            mz_values, intensity_values = spectrum.get_peaks()
            group.create_dataset("mz", data=np.array(mz_values, dtype=np.float16))
            group.create_dataset("intensity", data=np.array(intensity_values, dtype=np.float16))
        print(f"Die Daten wurden erfolgreich in '{hdf5_filename}' gespeichert.")


def mzml_to_array(file_path, mZ_totlist):
    if not isinstance(file_path, str):
        raise TypeError("file_path must be a string.")
    if not os.path.isfile(file_path):
        raise ValueError("file_path must be a valid file path.")

    exp = MSExperiment()
    MzMLFile().load(file_path, exp)
    chromatogram = []

    

    for spectrum in exp:
        if spectrum.getMSLevel() == 1:
            mz, intensity = spectrum.get_peaks()
            

            full_intensity = np.zeros(len(mZ_totlist))
            bins = np.digitize(mz, mZ_totlist)
            full_intensity[bins - 1] = intensity

            chromatogram.append(full_intensity.tolist())
    return np.array(chromatogram)



# Funktion zum Laden der HDF5-Datei und Anzeigen der Daten
def load_hdf5(hdf5_filename):
    data = []
    with h5py.File(hdf5_filename, "r") as f:
        for rt in f.keys():
            group = f[rt]
            mz_values = group["mz"][:]
            intensity_values = group["intensity"][:]
            for mz, intensity in zip(mz_values, intensity_values):
                data.append((rt, mz, intensity))
                if len(data) >= 1000000:  # Adjust the chunk size as needed
                    yield pd.DataFrame(data, columns=["RT", "mz", "intensity"])
                    data = []
        if data:
            yield pd.DataFrame(data, columns=["RT", "mz", "intensity"])

def load_hdf5_to_dask(hdf5_filename):
    dfs = []
    for chunk in load_hdf5(hdf5_filename):
        dfs.append(dd.from_pandas(chunk, npartitions=4))
    return dd.concat(dfs)

def convert_d_to_mzml(path):
    mzml_path = os.path.join(path, "mzml/")
    for file in os.listdir(path):
        if file.endswith(".D"):
            mzml_file = os.path.join(mzml_path, file.replace(".D", ".mzML"))
            if not os.path.exists(mzml_file):
                command = f"msconvert {os.path.join(path, file)} -o {mzml_path} --mzML --filter \"peakPicking true 1-\""
                subprocess.run(command, shell=True, check=True)


def mzml_to_array(file_path):
    if not isinstance(file_path, str):
        raise TypeError("file_path must be a string.")
    if not os.path.isfile(file_path):
        raise ValueError("file_path must be a valid file path.")

    exp = MSExperiment()
    MzMLFile().load(file_path, exp)
    chromatogram = dict()

    for spectrum in exp.getSpectra():
        rt = spectrum.getRT()/60
        mz_values, intensity_values = spectrum.get_peaks()
        chromatogram[rt] = (mz_values, intensity_values)
    
    return chromatogram


def search_for_mz(chromatogram, decimal_place , number):
    '''
    Search for m/z values which have as the value a as first decimal place
    '''
    new_chromatogram = dict()
    for rt, (mz_values, intensity_values) in chromatogram.items():
        new_mz_values = []
        new_intensity_values = []
        for mz, intensity in zip(mz_values, intensity_values):
            if round(round(mz, decimal_place)-round(mz,decimal_place-1),decimal_place) == number/(10**decimal_place):
                new_mz_values.append(mz)
                new_intensity_values.append(intensity)
        new_chromatogram[rt] = (new_mz_values, new_intensity_values)
    return new_chromatogram

def plot_dict(chromatogram):
    '''
    Plot chromatogram from dictionary. X-Values are the keys of the dictionary (retention time) and the Y-values are the sum of the intensity values.
    '''
    import matplotlib.pyplot as plt
    # Group the data by the retention time
    all_rt = []
    sum_intensity = []
    for rt, (_, intensity_values) in chromatogram.items():
        all_rt.append(rt)
        sum_intensity.append(sum(intensity_values))
    # Create a figure
    plt.figure(figsize=(10, 6))
    # Plot the retention time against the intensity
    plt.plot(all_rt, sum_intensity)
    # Add labels and a legend
    plt.xlabel("RT")
    plt.ylabel("Intensity")
    # Show the plot
    plt.show()

def get_all_unique_mz(chromatogram):
    '''
    Get all unique m/z values from a chromatogram
    '''
    all_mz = []
    for _, (mz_values, _) in chromatogram.items():
        all_mz.extend(mz_values)
    return set(all_mz)

# Define path to the mzML file
#PATH = 'F:/Documents/MasterArbeit/QTOF_DATA/'
#EXPORT_PATH = 'F:/Documents/MasterArbeit/ACLS_Masterarbeit/SideProject/Data/'
PATH = 'C:/Users/wilv/Documents/Masterarbeit/QTOF-DATA/'
EXPORT_PATH = 'C:/Users/wilv/Documents/Masterarbeit/ACLS_Masterarbeit/SideProject/Data/'
# t1 = t.time()
# convert_d_to_mzml(PATH)
# t2 = t.time()
# print(f"Time: {t2-t1} s")

# Time to convert .D files to .mzML files: 400 s on WorkPC


FILENAME = 'Larve_washed_02.mzML'
# Beispiel: Verwende die Funktion
#mzml_to_hdf5(PATH+"mzml/"+FILENAME, EXPORT_PATH+"daten.h5")


# import mzML to Array
t1 = t.time()
imported = mzml_to_array(PATH+"mzml/"+FILENAME)
t2 = t.time()
#check how many item the dictionary has
print(len(imported))
#check the last keys of the dictionary
print(list(imported.keys())[-5:])
#check the used space of the dictionary
print(imported.__sizeof__())

print(f"Time: {t2-t1} s")

t1 = t.time()
cleared = search_for_mz(imported, 2, 1)
t2 = t.time()

#check how many item the dictionary has
print(len(cleared))
#check the last keys of the dictionary
print(list(cleared.keys())[-5:])
#check the used space of the dictionary
print(cleared.__sizeof__())

print(f"Time: {t2-t1} s")

plot_dict(imported)
plot_dict(cleared)

# Get all unique m/z values
all_mz = get_all_unique_mz(imported)
# take the first value
print(round(round(list(all_mz)[0],2)-round(list(all_mz)[0],1),2))






