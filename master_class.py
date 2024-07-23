import os
import subprocess
import numpy as np
import pandas as pd
import pyopenms
import copy as cp
import math
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def frontslash_to_backslash(string):
    return string.replace('/', '\\')

class Data_Preparation:
    def __init__(self, path):
        self.path = path
        self.mzml_path = self.path + "/mzml"
        if not os.path.exists(self.mzml_path):
            os.mkdir(self.mzml_path)
        self.msdial_path = self.path + "/MSDIAL"
        if not os.path.exists(self.msdial_path):
            os.mkdir(self.msdial_path)
        self.csv_path = self.path + "/CSV"
        if not os.path.exists(self.csv_path):
            os.mkdir(self.csv_path)

        self.mZ_totlist = np.round(np.arange(20, 400.1, 0.1), 1)

    ####################################################################################################################
    def convert_d_to_mzml(self):
        for file in os.listdir(self.path):
            if file.endswith(".D"):
                #check if file .D is already converted to .mzml
                if not os.path.exists(self.mzml_path + '/' + file.replace(".D", ".mzML")):
                    command = f"msconvert {self.path+file} -o {self.mzml_path} --mzML"
                    subprocess.run(command, shell=True)


    def get_name_mzml_files(self):
        return os.listdir(self.mzml_path)



    ####################################################################################################################
    def interact_with_msdial(self, msdial_path, param_file_name, type="gcms"):
        # MsdialConsoleApp.exe <analysisType> -i <input folder> -o <output folder> -m <method file> -p (option)
        self.msdial_params = self.path + '/' + param_file_name + '.txt'
        self.msdial_consol_app = msdial_path
        self.type = type

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
        pyopenms.MzMLFile().load(self.mzml_path+'/'+mzml_file, exp)
        # a full list of retention times
        rt = []
        for spectrum in exp:
            if spectrum.getMSLevel() == 1:
                rt.append(spectrum.getRT()/60)
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
    def get_list_of_chromatograms(self, NAMES, Type = "FromMzml"):
        if Type == "FromMzml":
            self.chromatograms = dict()
            for name in NAMES:
                self.chromatograms[name] = self.mzml_to_array(self.mzml_path + '/' + name)
        elif Type == "FromNPY":
            self.chromatograms = np.load(NAMES, allow_pickle=True).item()
        else:
            raise ValueError("Type must be either 'FromMzml' or 'FromNPY'")

        return self.chromatograms

    ####################################################################################################################

    def get_chromatogram(self, name):
        return self.chromatograms[name]

    ####################################################################################################################

    def plot_chromatogram(self, name):
        rt = self.get_retention_time()
        if isinstance(name, list):
            for i in name:
                plt.figure(figsize=(12, 5))
                plt.plot(rt,np.sum(self.chromatograms[i], axis = 1))
                plt.xlabel('Retention Time')
                plt.ylabel('Intensity')
                # hide the right and top spines
                plt.gca().spines['right'].set_visible(False)
                plt.gca().spines['top'].set_visible(False)
                plt.show()
        elif isinstance(name, str):
            plt.figure(figsize=(12, 5))
            plt.plot(rt,np.sum(self.chromatograms[name], axis = 1))
            plt.xlabel('Retention Time')
            plt.ylabel('Intensity')
            # hide the right and top spines
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['top'].set_visible(False)
            plt.show()


    ####################################################################################################################

    def parse_msp_alignment_compounds(self, msp_file_path):
        self.alignment_compound_list = {}
        with open(msp_file_path, 'r') as file:
            spectra = []
            Flag = False
            for line in file:
                line = line.strip()
                if not line:
                    Flag = False
                    self.alignment_compound_list[name] = spectra
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
        mz_compresed = np.arange(20, 401, 1)

        for i in self.alignment_compound_list.keys():
            # Erstellen eines Dictionaries aus den gegebenen Daten für schnellen Zugriff
            data_dict = {int(mz): intensity for mz, intensity in self.alignment_compound_list[i]}

            # Durchlaufen der Ziel-Liste und Überprüfen auf vorhandene m/z-Werte
            result = []
            for mz in mz_compresed:
                intensity = data_dict.get(mz, 0)  # Falls der m/z-Wert nicht gefunden wird, setze die Intensität auf 0
                result.append([mz, intensity])

            # Umwandeln der resultierenden Liste in ein numpy-Array
            self.alignment_compound_list[i] = np.array(result)


    ####################################################################################################################

    def get_decomposition_compound_list(self):
        return self.alignment_compound_list

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

    ####################################################################################################################

    def compression_of_spectra(self):
        '''
        This function compresses the spectra to a smaller size from 0.1 to 1.0
        :return:
        '''
        for i in self.chromatograms.keys():
            chroma = self.chromatograms[i]
            # iterate over the chromatogram and compress the spectra
            # add first 5 elements, then the sum of the next 10 elements until the last 5 elements
            compressed_chroma = np.sum(chroma[:,0:7], axis=1)
            for j in range(7, np.shape(chroma)[1]-3, 10):
                summed_columns = np.sum(chroma[:,j:j+10], axis=1)
                compressed_chroma = np.vstack((compressed_chroma, summed_columns))
            #compressed_chroma = np.vstack((compressed_chroma, np.sum(chroma[:,-5:], axis=1)))

            compressed_chroma = np.transpose(compressed_chroma)

            self.chromatograms[i] = compressed_chroma


    ####################################################################################################################

    def get_rt_of_alignment_compounds(self, name_chromatogram):
        # set the RI values of the alignment compounds
        #RI = [1185, 1289, 1647, 2450]
        RI = [1391, 1647, 2414]
        # get the retention time and calculating the RI-factor for the hole chromatogram
        rt = self.get_retention_time()
        RI_ist = 4.1643 * rt ** 2 - 52.126 * rt + 1322.4
        #RI_ist = 3.8012 * rt ** 2 - 40.943 * rt + 1243
        # initialize the rt_index list
        rt_index = []
        pre_match_matrix_list = []
        match_matrix_list = []
        # set the wight factor for the macht_factor_RI
        k = 150
        j = 0
        for i in self.alignment_compound_list.keys():
            chroma = cp.copy(self.chromatograms[name_chromatogram])
            chroma = self.normalize_spectra(chroma)
            alignment_spectra = cp.copy(self.alignment_compound_list[i][:,1])
            match_matrix = np.dot(chroma, np.transpose(alignment_spectra))/10**4
            # add an RI-factor to the match_matrix to get the best match
            RI_i = RI[j]
            eq = RI_ist/RI_i
            match_factor_RI = np.exp(-k*(eq-1)**2)
            pre_match_matrix_list.append(match_matrix)
            match_matrix = match_matrix * match_factor_RI
            match_matrix_list.append(match_matrix)
            rt_index.append(np.argmax(match_matrix))
            j += 1
        # sort the rt_index
        #rt_index = np.sort(rt_index)
        #rt_return = rt[rt_index]

        return rt_index, match_matrix_list, pre_match_matrix_list


    ####################################################################################################################

    def normalize_spectra(self, array):
        # if the array is n 2D array normalize each column
        if len(np.shape(array)) > 1:
            for i in range(np.shape(array)[0]):
                array[i] = self.normalize_array(array[i])
        else:
            array = self.normalize_array(array)
        return array

    def normalize_array(self, array):
        # Normalize the array to the interval [0, 1000]
        #array = array - np.min(array)
        array = array / np.max(array)
        array = array * 999
        return array

    ####################################################################################################################

    def set_comparison_chromatogram(self, name_chromatogram):
        self.rt_comparison_chroma,_ = self.get_rt_of_alignment_compounds(name_chromatogram)


    ####################################################################################################################
    def aligned_chromatograms(self, name_chromatogram):
        # get the retention time
        rt = self.get_retention_time()
        # get the chromatogram
        chroma = self.chromatograms[name_chromatogram]

        # rt of the alignment compounds
        rt_index_alignment_comps,_ = self.get_rt_of_alignment_compounds(name_chromatogram)

        rt_index_comparsion_chroma = self.rt_comparison_chroma

        for i in range(len(rt_index_alignment_comps)+1):

            if i == 0:
                # get the part of the chromatogram from the beginning to the first alignment compound, without it self
                part_chroma = chroma[:rt_index_alignment_comps[i]]
                part_rt = rt[:rt_index_alignment_comps[i]]
                self.get_fit_parameters(part_rt, part_chroma)
            elif i == len(rt_index_alignment_comps):
                # get the part of the chromatogram from the last alignment compound to the end
                part_chroma = chroma[rt_index_alignment_comps[i]-1:]
            else:
                # get the part of the chromatogram between two alignment compounds
                part_chroma = chroma[rt_index_alignment_comps[i-1]:rt_index_alignment_comps[i]]




    ####################################################################################################################

    def get_fit_parameters(self,rt,chroma):

        mz = np.arange(20, 401, 1)
        rt, mz = np.meshgrid(rt, mz, copy=False)
        rt = rt.flatten()
        mz = mz.flatten()
        chroma = chroma.flatten()

        poly = PolynomialFeatures(degree=200)

        input_pts = np.stack([rt, mz]).T
        #assert(input_pts.shape == (651, 12))
        in_features = poly.fit_transform(input_pts)

        print(f"Shape of input_pts (X): {input_pts.shape}")
        print(f"Shape of chroma (y): {chroma.shape}")
        # Linear regression
        model = LinearRegression(fit_intercept=False)
        model.fit(in_features, chroma)

        # Display coefficients
        print(dict(zip(poly.get_feature_names_out(), model.coef_.round(4))))

        # Check fit
        print(f"R-squared: {model.score(poly.transform(input_pts), chroma):.3f}")

        # Make predictions
        Z_predicted = model.predict(poly.transform(input_pts))


