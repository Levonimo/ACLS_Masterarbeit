import subprocess
import os
import pyopenms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def convert_d_to_mzml(d_file_path, mzml_file_path):
    command = f"msconvert {d_file_path} -o {mzml_file_path} --mzML"
    subprocess.run(command, shell=True)

def convert_all_d_to_mzml(d_folder_path):
    mzml_folder_path = d_folder_path + "/mzml"
    if not os.path.exists(mzml_folder_path):
        os.mkdir(mzml_folder_path)

    for file in os.listdir(d_folder_path):
        if file.endswith(".D"):
            convert_d_to_mzml(os.path.join(d_folder_path, file), os.path.join(mzml_folder_path))


def interact_with_msdial(file_path, msdial_path, param_file_name , type="gcms"):
    # MsdialConsoleApp.exe <analysisType> -i <input folder> -o <output folder> -m <method file> -p (option)
    mzml_file_path = file_path + '/mzml'
    msdial_params = file_path + '/' + param_file_name + '.txt'
    output_folder = file_path + '/MSDIAL'

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # Replace all frontslashes with backslashes
    msdial_path = frontslash_to_backslash(msdial_path)
    mzml_file_path = frontslash_to_backslash(mzml_file_path)
    output_folder = frontslash_to_backslash(output_folder)
    msdial_params = frontslash_to_backslash(msdial_params)

    command = f"{msdial_path} {type} -i {mzml_file_path} -o {output_folder} -m {msdial_params}"
    subprocess.run(command, shell=True)

def frontslash_to_backslash(string):
    return string.replace('/', '\\')

def mzml_to_dict(mzml_file_path):
    exp = pyopenms.MSExperiment()
    pyopenms.MzMLFile().load(mzml_file_path, exp)

    Chromatogramm = dict()
    mZ_totlist = np.arange(20, 400.1, 0.1)
    mZ_totlist = np.round(mZ_totlist, 1)

    for spectrum in exp:
        if spectrum.getMSLevel() == 1:
            rt = spectrum.getRT() / 60
            mz, intensity = spectrum.get_peaks()
            mz = np.round(mz, 1)

            Full_intensity = np.zeros(len(mZ_totlist))
            bins = np.digitize(mz, mZ_totlist)
            Full_intensity[bins - 1] = intensity

            Chromatogramm[rt] = Full_intensity
    return Chromatogramm

def mzml_to_array(mzml_file_path):
    exp = pyopenms.MSExperiment()
    pyopenms.MzMLFile().load(mzml_file_path, exp)

    Chromatogramm = []
    rt_list = []
    mZ_totlist = np.arange(20, 400.1, 0.1)
    mZ_totlist = np.round(mZ_totlist, 1)

    for spectrum in exp:
        if spectrum.getMSLevel() == 1:
            rt = spectrum.getRT() / 60
            mz, intensity = spectrum.get_peaks()
            mz = np.round(mz, 1)

            Full_intensity = np.zeros(len(mZ_totlist))
            bins = np.digitize(mz, mZ_totlist)
            Full_intensity[bins - 1] = intensity

            Chromatogramm.append(Full_intensity.tolist())
            rt_list.append(rt)
    return np.array(Chromatogramm) , mZ_totlist, np.array(rt_list)

def standardize_chromatograms(chromatograms):
    max_length = max([len(chromatogram) for chromatogram in chromatograms.values()])
    for chromatogram in chromatograms.values():
        chromatogram.resize(max_length)
    return chromatograms

def plot_chromatograms(chromatograms):
    x = []
    y = []
    for retention_time, intensities in chromatograms.items():
        x.append(retention_time)
        y.append(sum(intensities))
    x = np.array(x)
    y = np.array(y)

    # Erstellen Sie eine neue Figur
    fig = plt.figure()

    # Fügen Sie eine neue Achse hinzu
    ax = fig.add_subplot(111)
    ax.plot(x, y, 'r-')
    ax.set_xlabel('Retention Time (min)')
    ax.set_ylabel('Total Ion Current')
    plt.show()


def extract_peaks(chromatogram, height_threshold):
    # Calculate the Total Ion Chromatogram
    tic = np.array([sum(intensities) for intensities in chromatogram.values()])

    # Calculate the derivative of the TIC
    derivative = np.gradient(tic)

    # Identify the peaks
    peaks = {}
    for i, (rt, intensities) in enumerate(chromatogram.items()):
        if i == 0:  # Skip the first point because there's no previous point to compare with
            continue
        if derivative[i-1] > 0 and derivative[i] < 0:
            peaks[rt] = intensities
    heighest_peak = {}
    for rt, intensities in peaks.items():
        if np.sum(intensities) > height_threshold:
            heighest_peak[rt] = intensities

    return heighest_peak

def convert_msdial_to_csv(folder_path):
    # Liste aller Dateien im Ordner
    files = os.listdir(folder_path+'/MSDIAL')

    # Filtere nur .msdial Dateien
    msdial_files = [file for file in files if file.endswith('.msdial')]

    for file in msdial_files:
        # Lese die .msdial Datei
        data = pd.read_csv(os.path.join(folder_path+'/MSDIAL', file), sep='\t')

        # Erstelle den Namen der .csv Datei
        csv_file_name = file.replace('.msdial', '.csv')

        # Speichere die Daten als .csv
        data.to_csv(os.path.join(folder_path+'/MSDIAL', csv_file_name), index=False)

    print(f'Converted {len(msdial_files)} .msdial files to .csv')


def plot_a_list_of_chromatograms_array(chromatograms,rt):
    '''
    Plots a dict of chromatograms overlapping each other
    :param chromatograms:
    :return:
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if isinstance(chromatograms, dict):
        for chromatogram in chromatograms.values():
            ax.plot(rt, np.sum(chromatogram, axis=1))
        # add an legenden to the plot with the names of the chromatograms
        ax.legend(chromatograms.keys(), frameon=False)
    elif isinstance(chromatograms[0], float):
        ax.plot(rt, chromatograms)
    elif isinstance(chromatograms, list):
        for chromatogram in chromatograms:
            ax.plot(rt, np.sum(chromatogram, axis=1))


    # set picture ratio to 15:5
    fig.set_size_inches(15, 5)
    # minimize the white space around the plot
    plt.tight_layout( pad=2)
    # disable the box around the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # set the labels for the x and y axis
    ax.set_xlabel('Retention Time (min)')
    ax.set_ylabel('Total Ion Current')
    plt.show()

def import_list_of_mzml_files(PATH, NAMES):
    chromatograms = dict()
    for name in NAMES:
            chromatograms[name] = mzml_to_array(PATH + name)[0]

    rt = mzml_to_array(PATH + NAMES[0])[2]
    return chromatograms ,rt


def mean_of_chromatograms(chromatograms):
    # Calculate the total ion current for each chromatogram
    tic = {name: np.sum(chromatogram, axis=1) for name, chromatogram in chromatograms.items()}

    # Calculate the mean of the total ion current
    mean_tic = np.mean([intensities for intensities in tic.values()], axis=0)

    return mean_tic


def mean_of_dict(dict_of_arrays):
    # Initialize a variable to store the sum of arrays
    sum_of_arrays = None

    # Iterate over the dictionary values (which are arrays)
    for array in dict_of_arrays.values():
        if sum_of_arrays is None:
            sum_of_arrays = array
        else:
            sum_of_arrays += array

    # Calculate the mean by dividing the sum by the number of arrays
    mean_array = sum_of_arrays / len(dict_of_arrays)

    return mean_array


def find_common_peaks_in_csv_files(folder_path):
    # Liste aller Dateien im Ordner
    files = os.listdir(folder_path + '/MSDIAL')

    # Filtere nur .csv Dateien
    csv_files = [file for file in files if file.endswith('.csv')]

    # Lese die erste Datei
    data = pd.read_csv(os.path.join(folder_path + '/MSDIAL', csv_files[0]))

    # Erstelle eine Liste aller Peaks
    filtered_data = data[data['Model ion height'] > 10e5]
    peaks = set(filtered_data['Model ion mz'])

    # Iteriere über die restlichen Dateien und finde die Schnittmenge
    # an Peaks in allen Dateien (common peaks) heraus (intersection)
    # welche zudem eine 'Model ion height' von grösser 10e5 haben und gleiche Retentionszeit haben
    for file in csv_files[1:]:
        data = pd.read_csv(os.path.join(folder_path + '/MSDIAL', file))
        filtered_data = data[data['Model ion height'] > 10e5]
        peaks = peaks.intersection(set(filtered_data['Model ion mz']))



    print(f'Found {len(peaks)} common peaks in all files')
    return peaks