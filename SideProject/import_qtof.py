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
                command = f"msconvert {os.path.join(path, file)} -o {mzml_path} --mzML"
                subprocess.run(command, shell=True, check=True)

# Function that iterate over the dask dataframe and set all intensity values to zero were the mz values 
# have a certain decimal place. It should work as a mass defect filter.
# Possible inputs: dask dataframe, decimal place, mass defect threshold
def mass_defect_filter(df, decimal_place, threshold):
    def filter_row(row):
        mz = row["mz"]
        mass_defect = mz - round(mz, decimal_place)
        if mass_defect < threshold:
            row["intensity"] = 0
        return row

    return df.map_partitions(lambda df: df.apply(filter_row, axis=1))

# Function that plots the filtered data with matplotlib as a chromatogram over the retention time
def plot_chromatogram(df):
    import matplotlib.pyplot as plt
    # Group the data by the retention time
    grouped = df.groupby("RT")
    # Create a figure
    plt.figure(figsize=(10, 6))
    # Iterate over the groups
    for name, group in grouped:
        # Plot the retention time against the intensity
        plt.plot(group["mz"], group["intensity"], label=f"RT: {name}")
    # Add labels and a legend
    plt.xlabel("m/z")
    plt.ylabel("Intensity")
    plt.legend()
    # Show the plot
    plt.show()

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



# Import the mzML file and convert it to an array
mzml_file = os.path.join(PATH, "mzml", FILENAME)

# Define the total m/z list
mZ_totlist = np.arange(20, 500.00001, 0.00001)
print(len(mZ_totlist))
t1 = t.time()
chromatogram = mzml_to_array(mzml_file, mZ_totlist)
t2 = t.time()
print(f"Time: {t2-t1} s")

# Save the chromatogram as a h5 file
with h5py.File(EXPORT_PATH+"chromatogram.h5", "w") as f:
    f.create_dataset("chromatogram", data=chromatogram)
    print(f"Die Daten wurden erfolgreich in '{EXPORT_PATH+'chromatogram.h5'}' gespeichert.")



# t1 = t.time()
# # Verwende die Funktion zum Laden der HDF5-Datei
# df = load_hdf5_to_dask(EXPORT_PATH+"daten.h5")
# t2 = t.time()
# print(f"Time: {t2-t1} s")

# t1 = t.time()
# # Verwende die Funktion zum Filtern der Daten
# df_filtered = mass_defect_filter(df, 2, 0.01)
# t2 = t.time()
# print(f"Time: {t2-t1} s")

# # Verwende die Funktion zum Plotten des Chromatogramms
# plot_chromatogram(df_filtered.compute())
