import os
import subprocess
import numpy as np
import pandas as pd
import pyopenms
import copy as cp
import math
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from Warping import correlation_optimized_warping as COW

def frontslash_to_backslash(string):
    if not isinstance(string, str):
        raise TypeError("Input must be a string.")
    return string.replace('/', '\\')

class DataPreparation:
    def __init__(self, path):
        if not isinstance(path, str):
            raise TypeError("Path must be a string.")
        if not os.path.isdir(path):
            raise ValueError("Path must be a valid directory.")

        self.path = path
        self.mzml_path = os.path.join(self.path, "mzml")
        os.makedirs(self.mzml_path, exist_ok=True)

        self.msdial_path = os.path.join(self.path, "MSDIAL")
        os.makedirs(self.msdial_path, exist_ok=True)

        self.csv_path = os.path.join(self.path, "CSV")
        os.makedirs(self.csv_path, exist_ok=True)

        self.mZ_totlist = np.round(np.arange(35, 400.1, 0.1), 1)

        self.convert_d_to_mzml()

    def convert_d_to_mzml(self):
        for file in os.listdir(self.path):
            if file.endswith(".D"):
                mzml_file = os.path.join(self.mzml_path, file.replace(".D", ".mzML"))
                if not os.path.exists(mzml_file):
                    command = f"msconvert {os.path.join(self.path, file)} -o {self.mzml_path} --mzML"
                    subprocess.run(command, shell=True, check=True)

    def get_file_names(self):
        files = os.listdir(self.mzml_path)
        #remove the .mzML ending
        files = [file.replace('.mzML','') for file in files]
        return files
    '''
    def interact_with_msdial(self, msdial_path, param_file_name, type="gcms"):
        if not isinstance(msdial_path, str):
            raise TypeError("msdial_path must be a string.")
        if not os.path.isfile(msdial_path):
            raise ValueError("msdial_path must be a valid file path.")
        if not isinstance(param_file_name, str):
            raise TypeError("param_file_name must be a string.")
        if not isinstance(type, str):
            raise TypeError("type must be a string.")
        if type not in ["gcms", "lcms"]:
            raise ValueError("type must be 'gcms' or 'lcms'.")

        self.msdial_params = os.path.join(self.path, f'{param_file_name}.txt')
        self.msdial_consol_app = msdial_path
        self.type = type

        # Replace all frontslashes with backslashes
        msdial_app = frontslash_to_backslash(self.msdial_consol_app)
        mzml_folder = frontslash_to_backslash(self.mzml_path)
        msdial_folder = frontslash_to_backslash(self.msdial_path)
        msdial_params_file = frontslash_to_backslash(self.msdial_params)

        command = f"{msdial_app} {self.type} -i {mzml_folder} -o {msdial_folder} -m {msdial_params_file}"
        subprocess.run(command, shell=True, check=True)

    def convert_msdial_to_csv(self):
        msdial_files = [file for file in os.listdir(self.msdial_path) if file.endswith('.msdial')]
        if not msdial_files:
            raise ValueError("No .msdial files found in the MSDIAL directory.")

        for file in msdial_files:
            data = pd.read_csv(os.path.join(self.msdial_path, file), sep='\t')
            csv_file_name = file.replace('.msdial', '.csv')
            data.to_csv(os.path.join(self.csv_path, csv_file_name), index=False)

        print(f'Converted {len(msdial_files)} .msdial files to .csv')
    '''
    def get_retention_time(self):
        exp = pyopenms.MSExperiment()
        mzml_files = os.listdir(self.mzml_path)
        if not mzml_files:
            raise ValueError("No mzML files found in the mzml directory.")
        
        mzml_file = mzml_files[0]
        pyopenms.MzMLFile().load(os.path.join(self.mzml_path, mzml_file), exp)
        
        rt = [spectrum.getRT() / 60 for spectrum in exp if spectrum.getMSLevel() == 1]
        return np.array(rt)

    def get_mz_list(self):
        return self.mZ_totlist

    def mzml_to_array(self, file_path):
        if not isinstance(file_path, str):
            raise TypeError("file_path must be a string.")
        if not os.path.isfile(file_path):
            raise ValueError("file_path must be a valid file path.")

        exp = pyopenms.MSExperiment()
        pyopenms.MzMLFile().load(file_path, exp)
        chromatogram = []

        for spectrum in exp:
            if spectrum.getMSLevel() == 1:
                mz, intensity = spectrum.get_peaks()
                mz = np.round(mz, 1)

                full_intensity = np.zeros(len(self.mZ_totlist))
                bins = np.digitize(mz, self.mZ_totlist)
                full_intensity[bins - 1] = intensity

                chromatogram.append(full_intensity.tolist())
        chromatogram = self.compression_of_spectra(np.array(chromatogram))
        return np.array(chromatogram)


    def get_list_of_chromatograms(self, file, file_list = None):
        if not isinstance(file, str):
            raise TypeError("file must be a string.")
        if not isinstance(file_list, list) or not all(isinstance(name, str) for name in file_list):
            raise TypeError("file_list must be a list of strings.")
        if not file:
            file = "Chromatograms"

        if file_list:
            if not os.path.isfile(self.path+file+'.npy'):
                self.chromatograms = {}
                for name in file_list:
                    file_path = os.path.join(self.mzml_path, name+'.mzML')
                    if not os.path.isfile(file_path):
                        raise ValueError(f"{file_path} does not exist.")
                    self.chromatograms[name] = self.mzml_to_array(file_path)
                np.save(self.path+file+'.npy', self.chromatograms)
            else:
                np.load(self.path+file+'.npy', allow_pickle=True).item()
        else:
            if not os.path.isfile(self.path+file+'.npy'):
                raise ValueError(f"{file}.npy does not exist and no name list is given.")
            else:
                np.load(self.path+file+'.npy', allow_pickle=True).item()
        '''        
        if not isinstance(names, list) and not isinstance(names, str):
            raise TypeError("names must be a list of strings or a single string.")
        if not all(isinstance(name, str) for name in names) and not isinstance(names, str):
            raise TypeError("All elements in names must be strings.")
        if source_type not in ["FromMzml", "FromNPY"]:
            raise ValueError("source_type must be either 'FromMzml' or 'FromNPY'.")

        self.chromatograms = {}
        
        if source_type == "FromMzml":
            for name in names:
                file_path = os.path.join(self.mzml_path, name)
                if not os.path.isfile(file_path):
                    raise ValueError(f"{file_path} does not exist.")
                self.chromatograms[name] = self.mzml_to_array(file_path)
        elif source_type == "FromNPY":
            if not os.path.isfile(names):
                raise ValueError(f"{names} is not a valid file path.")
            self.chromatograms = np.load(names, allow_pickle=True).item()
        '''
        return self.chromatograms

    def get_chromatogram(self, name):
        if not isinstance(name, str):
            raise TypeError("name must be a string.")
        return self.chromatograms.get(name)

    def plot_chromatogram(self, name):
        if not isinstance(name, str) and not isinstance(name, list):
            raise TypeError("name must be a string or a list of strings.")
        if isinstance(name, list) and not all(isinstance(i, str) for i in name):
            raise TypeError("All elements in the list must be strings.")

        rt = self.get_retention_time()
        if isinstance(name, list):
            for i in name:
                plt.figure(figsize=(12, 5))
                plt.plot(rt, np.sum(self.chromatograms[i], axis=1))
                plt.xlabel('Retention Time')
                plt.ylabel('Intensity')
                plt.gca().spines['right'].set_visible(False)
                plt.gca().spines['top'].set_visible(False)
                plt.show()
        elif isinstance(name, str):
            plt.figure(figsize=(12, 5))
            plt.plot(rt, np.sum(self.chromatograms[name], axis=1))
            plt.xlabel('Retention Time')
            plt.ylabel('Intensity')
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['top'].set_visible(False)
            plt.show()

    def parse_msp_alignment_compounds(self, msp_file_path):
        if not isinstance(msp_file_path, str):
            raise TypeError("msp_file_path must be a string.")
        if not os.path.isfile(msp_file_path):
            raise ValueError("msp_file_path must be a valid file path.")

        self.alignment_compound_list = {}
        with open(msp_file_path, 'r') as file:
            spectra = []
            name = None
            for line in file:
                line = line.strip()
                if not line and name:
                    self.alignment_compound_list[name] = spectra
                    spectra = []
                elif line.startswith("Name:"):
                    name = line.split(": ", 1)[1]
                elif line.startswith("Num Peaks:"):
                    spectra = []
                elif name and spectra is not None:
                    mz_int_pair = [float(x) for x in line.split() if x]
                    spectra.append(mz_int_pair)

        mz_compressed = np.arange(20, 401, 1)

        for name, spectra in self.alignment_compound_list.items():
            data_dict = {int(mz): intensity for mz, intensity in spectra}
            intensity_list = [data_dict.get(mz, 0) for mz in mz_compressed]
            self.alignment_compound_list[name] = np.array(intensity_list)

    def get_alignment_compound_list(self):
        return self.alignment_compound_list

    def normalize_chromatogram(self, chromatogram):
        if not isinstance(chromatogram, np.ndarray):
            raise TypeError("chromatogram must be a numpy array.")
        if chromatogram.ndim != 2:
            raise ValueError("chromatogram must be a 2D numpy array.")

        norm_factor = np.sum(chromatogram, axis=1)
        norm_factor[norm_factor == 0] = 1  # Avoid division by zero
        return chromatogram / norm_factor[:, None]
    
    def compression_of_spectra(self, chromatogram):
        '''
        This function compresses the spectra to a bigger step size from 0.1 to 1
        :return:
        '''
        compressed_chroma = np.sum(chromatogram[:,0:7], axis=1)
        # iterate over the chromatogram and compress the spectra
        # add first 5 elements, then the sum of the next 10 elements until the last 5 elements
        for j in range(7, np.shape(chromatogram)[1]-3, 10):
            summed_columns = np.sum(chromatogram[:,j:j+10], axis=1)
            compressed_chroma = np.vstack((compressed_chroma, summed_columns))
        #compressed_chroma = np.vstack((compressed_chroma, np.sum(chroma[:,-5:], axis=1)))

        compressed_chroma = np.transpose(compressed_chroma)

        return compressed_chroma

    def warping(self, reference, target):
        warped_target, _ = COW(reference,target)
        return warped_target

    def perform_pca(self,chromatograms, n_components=10):
        """
        Führt eine Principal Component Analysis (PCA) auf einer Matrix von Gaschromatogrammen durch.

        Parameters:
        chromatograms (np.array): Die Eingabematrix von Gaschromatogrammen.
        n_components (int): Die Anzahl der Hauptkomponenten, die extrahiert werden sollen.

        Returns:
        Tuple: Tuple mit den folgenden Werten:
            - pca (PCA-Objekt): Das PCA-Objekt, das die Hauptkomponenten enthält.
            - scores (np.array): Die Scores der Daten in den neuen Hauptkomponenten.
            - loadings (np.array): Die Loadings (Ladungen) der Variablen.
            - explained_variance_ratio (np.array): Der Anteil der Varianz, die durch jede Hauptkomponente erklärt wird.
        """
        chromatograms = np.array([chromatograms[key] for key in chromatograms.keys()])
        # Standardisierung der Daten
        scaler = StandardScaler()
        chromatograms_std = scaler.fit_transform(chromatograms)

        # PCA durchführen
        pca = PCA(n_components=n_components)
        scores = pca.fit_transform(chromatograms_std)

        # Ladungen (Loadings) sind die Koeffizienten der linearen Kombinationen der ursprünglichen Variablen
        loadings = pca.components_.T

        # Anteil der erklärten Varianz
        explained_variance_ratio = pca.explained_variance_ratio_

        return pca, scores, loadings, explained_variance_ratio

    def PCA(self, Chromatograms):
        Chromatograms = np.array([Chromatograms[key] for key in Chromatograms.keys()])
        Chromatograms = Chromatograms.T
        #Chromatograms = Chromatograms - np.mean(Chromatograms, axis = 0)
        covariance_matrix = np.cov(Chromatograms.T)
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        # Calculate Loadings and Scores
        scores = np.dot(Chromatograms, eigenvectors)
        loadings = np.dot(scores, eigenvectors.T)
        return scores, loadings, eigenvalues, eigenvectors

