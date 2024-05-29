import os
import subprocess
import numpy as np
import pandas as pd
import pyopenms



def frontslash_to_backslash(string):
    return string.replace('/', '\\')

class Data_Preparation:
    def __init__(self, path):
        self.path = path
        self.mzml_path = self.path + "/mzml"
        self.msdial_path = self.path + "/MSDIAL"
        self.csv_path = self.path + "/CSV"

        self.mZ_totlist = np.round(np.arange(20, 400.1, 0.1), 1)
    ####################################################################################################################
    def convert_d_to_mzml(self):

        if not os.path.exists(self.mzml_path):
            os.mkdir(self.mzml_path)

        for file in os.listdir(self.path):
            if file.endswith(".D"):
                command = f"msconvert {self.path} -o {self.mzml_path} --mzML"
                subprocess.run(command, shell=True)




    ####################################################################################################################
    def interact_with_msdial(self, msdial_path, param_file_name, type="gcms"):
        # MsdialConsoleApp.exe <analysisType> -i <input folder> -o <output folder> -m <method file> -p (option)
        self.msdial_params = self.path + '/' + param_file_name + '.txt'
        self.msdial_consol_app = msdial_path
        self.type = type

        # Create the output folder if it doesn't exist
        if not os.path.exists(self.msdial_path):
            os.mkdir(self.msdial_path)

        # Replace all frontslashes with backslashes
        msdial_app = frontslash_to_backslash(self.msdial_consol_app)
        mzml_folder = frontslash_to_backslash(self.mzml_path)
        msdial_folder = frontslash_to_backslash(self.msdial_path)
        msdial_params_file = frontslash_to_backslash(self.msdial_params)

        command = f"{msdial_app} {self.type} -i {mzml_folder} -o {msdial_folder} -m {msdial_params_file}"
        subprocess.run(command, shell=True)

    ####################################################################################################################
    def convert_msdial_to_csv(self):
        # Liste aller Dateien im Ordner
        files = os.listdir(self.msdial_path)

        # Filtere nur .msdial Dateien
        msdial_files = [file for file in files if file.endswith('.msdial')]

        for file in msdial_files:
            # Lese die .msdial Datei
            data = pd.read_csv(os.path.join(self.msdial_path, file), sep='\t')

            # Erstelle den Namen der .csv Datei
            csv_file_name = file.replace('.msdial', '.csv')

            # Speichere die Daten als .csv
            data.to_csv(os.path.join(self.csv_path, csv_file_name), index=False)

        print(f'Converted {len(msdial_files)} .msdial files to .csv')


    ####################################################################################################################
    def get_retention_time(self):
        exp = pyopenms.MSExperiment()
        # get one name of a mzml files
        mzml_file = os.listdir(self.mzml_path)[0]
        pyopenms.MzMLFile().load(mzml_file, exp)
        rt = exp[0].getRT() / 60
        return np.array(rt)

    ####################################################################################################################
    def get_mz_list(self):
        return self.mZ_totlist

    ####################################################################################################################
    def mzml_to_array(self, FILE_PATH):
        exp = pyopenms.MSExperiment()
        pyopenms.MzMLFile().load(FILE_PATH, exp)
        Chromatogramm = []

        for spectrum in exp:
            if spectrum.getMSLevel() == 1:
                mz, intensity = spectrum.get_peaks()
                mz = np.round(mz, 1)

                Full_intensity = np.zeros(len(self.mZ_totlist))
                bins = np.digitize(mz, self.mZ_totlist)
                Full_intensity[bins - 1] = intensity

                Chromatogramm.append(Full_intensity.tolist())

        return np.array(Chromatogramm)

    ####################################################################################################################
    def get_list_of_chromatograms(self, NAMES):
        chromatograms = dict()
        for name in NAMES:
            chromatograms[name] = self.mzml_to_array(self.mzml_path + name)


        return chromatograms

    ####################################################################################################################

    def get_similar_peak(self):
        # compare each csv File with each other and return the similar peaks
        # the similar peaks are defined as peaks which are in the same retention time window
        # and have a similar "Model ion mz" value

        # Liste aller Dateien im Ordner
        files = os.listdir(self.csv_path)

        # Filtere nur .csv Dateien
        csv_files = [file for file in files if file.endswith('.csv')]

        # Erstelle ein leeres Dictionary
        similar_peaks = dict()

        # Iteriere über alle Dateien
        for file in csv_files:
            # Lese die Datei
            data = pd.read_csv(os.path.join(self.csv_path, file))

            # Extrahiere die Retentionszeit
            retention_time = data['Retention time']

            # Extrahiere die Model ion mz
            model_ion_mz = data['Model ion mz']

            # Iteriere über alle Retentionszeiten
            for rt, mz in zip(retention_time, model_ion_mz):
                # Erstelle ein Tupel aus Retentionszeit und Model ion mz
                peak = (rt, mz)

                # Wenn der Peak nicht im Dictionary ist, füge ihn hinzu
                if peak not in similar_peaks:
                    similar_peaks[peak] = [file]
                # Wenn der Peak schon im Dictionary ist, füge die Datei hinzu
                else:
                    similar_peaks[peak].append(file)

        return similar_peaks

    ####################################################################################################################

    def parse_msp_file(self, msp_file_path):
        self.decomposition_compound_list = {}
        with open(msp_file_path, 'r') as file:
            spectra = []
            Flag = False
            for line in file:
                line = line.strip()
                if not line:
                    Flag = False
                    self.decomposition_compound_list[name] = spectra
                    spectra = []
                elif line.startswith("Name:"):
                    name = line.split(": ", 1)[1]
                elif line.startswith("Num Peaks:"):
                    Flag = True
                elif Flag:
                    splited = line.split(";")
                    for i in splited:
                        if i == "":
                            continue
                        mz_int_pair =[float(x) for x in i.split()]
                        spectra.append(mz_int_pair)

    ####################################################################################################################

    def get_decomposition_compound_list(self):
        return self.decomposition_compound_list

    ####################################################################################################################

    def get_spectra_from_csv(self):
        # Liste aller Dateien im Ordner
        files = os.listdir(self.csv_path)

        # Filtere nur .csv Dateien
        csv_files = [file for file in files if file.endswith('.csv')]

        # Erstelle ein leeres Dictionary
        self.spectra_from_csv = {}
        for i in csv_files:
            rt_spectra = []
            csv_file = pd.read_csv("F:/Documents/MasterArbeit/MSDIAL/" + i)
            for rt, spectra in zip(csv_file['RT(min)'], csv_file['Spectrum']):
                array = [[np.round(float(j), 1) for j in i] for i in [i.split(':') for i in spectra.split()]]
                for compound in self.decomposition_compound_list.values():
                    print(compound)


                rt_spectra.append([rt, array])
            self.spectra_from_csv[i] = rt_spectra
