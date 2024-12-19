import os
import subprocess
import numpy as np
import pandas as pd
import pyopenms as oms
import concurrent.futures
import multiprocessing


def convert_file(Dfile, path, mzml_path):
    # Hier bauen Sie den Befehl zum Umwandeln der Datei
    
    command = f"msconvert {os.path.join(path, Dfile)} -o {os.path.join(mzml_path)} --mzML --filter \"peakPicking true 1-\""
    try:
        subprocess.run(command, check=True)
        print(f'{Dfile} erfolgreich umgewandelt.')
    except subprocess.CalledProcessError as e:
        print(f'Fehler beim Umwandeln von {Dfile}: {e}')
    



def frontslash_to_backslash(string):
    if not isinstance(string, str):
        raise TypeError("Input must be a string.")
    return string.replace('/', '\\')

class DataPreparation:
    def __init__(self, path):
        self.chromatograms = {}
        if not isinstance(path, str):
            raise TypeError("Path must be a string.")
        if not os.path.isdir(path):
            raise ValueError("Path must be a valid directory.")

        self.path = path
        self.mzml_path = os.path.join(self.path, "mzml")
        os.makedirs(self.mzml_path, exist_ok=True)

        self.mZ_totlist = np.round(np.arange(35, 400.1, 0.1), 1)

        self.convert_d_to_mzml()

    # def convert_d_to_mzml(self):
    #     for file in os.listdir(self.path):
    #         if file.endswith(".D"):
    #             mzml_file = os.path.join(self.mzml_path, file.replace(".D", ".mzML"))
    #             if not os.path.exists(mzml_file):
    #                 command = f"msconvert {os.path.join(self.path, file)} -o {self.mzml_path} --mzML --filter \"peakPicking true 1-\""
    #                 subprocess.run(command, shell=True, check=True)


    def convert_d_to_mzml(self):
        files = [file for file in os.listdir(self.path) if file.endswith(".D")]

        mzml_file = [file.replace(".D", ".mzML") for file in files]

        mzml_file = [file for file in mzml_file if not os.path.exists(os.path.join(self.mzml_path, file))]

        files_to_process = [file for file in files if file.replace(".D", ".mzML") in mzml_file]
        print(multiprocessing.cpu_count())
        max_workers = 4

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Aufgaben an den Executor übergeben
            futures = [executor.submit(convert_file, Dfile, self.path, self.mzml_path) for Dfile in files_to_process]

            for future in concurrent.futures.as_completed(futures):
                future.result()




    def get_file_names(self):
        files = os.listdir(self.mzml_path)
        #remove the .mzML ending
        files = [file.replace('.mzML','') for file in files if file.endswith('.mzML')]
        return files
    
    def get_retention_time(self):
        exp = oms.MSExperiment()
        mzml_files = os.listdir(self.mzml_path)
        if not mzml_files:
            raise ValueError("No mzML files found in the mzml directory.")
        
        mzml_file = mzml_files[0]
        oms.MzMLFile().load(os.path.join(self.mzml_path, mzml_file), exp)
        
        rt = [spectrum.getRT() / 60 for spectrum in exp if spectrum.getMSLevel() == 1]
        return np.array(rt)

    def get_mz_list(self):
        return self.mZ_totlist

    def mzml_to_array(self, file_path):
        if not isinstance(file_path, str):
            raise TypeError("file_path must be a string.")
        if not os.path.isfile(file_path):
            raise ValueError("file_path must be a valid file path.")
        print(file_path)
        exp = oms.MSExperiment()
        oms.MzMLFile().load(file_path, exp)
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
            raise TypeError("file must be a string or an .npy-File.")
        if file_list:
            if not isinstance(file_list, list) or not all(isinstance(name, str) for name in file_list):
                raise TypeError("file_list must be a list of strings.")
        if not file:
            file = "Chromatograms"
        if file.endswith('.npy'):
            # strip the ending
            file = file.replace('.npy','')
        

        if file_list:
            if not os.path.isfile(self.path+ '/' +file+'.npy'):
                for name in file_list:
                    file_path = os.path.join(self.mzml_path, name+'.mzML')
                    if not os.path.isfile(file_path):
                        raise ValueError(f"{file_path} does not exist.")
                    self.chromatograms[name] = self.mzml_to_array(file_path)
                np.save(self.path+ '/' +file+'.npy', self.chromatograms)
            else:
                self.chromatograms = np.load(self.path+ '/' +file+'.npy', allow_pickle=True).item()
        else:
            if not os.path.isfile(self.path+ '/' +file+'.npy'):
                raise ValueError(f"{file} does not exist and no name list is given.")
            else:
                self.chromatograms = np.load(self.path+ '/' +file+'.npy', allow_pickle=True).item()

        return self.chromatograms



    def get_chromatogram(self, name):
        if not isinstance(name, str):
            raise TypeError("name must be a string.")
        return self.chromatograms.get(name)
    



    # def plot_chromatogram(self, name):
    #     if not isinstance(name, str) and not isinstance(name, list):
    #         raise TypeError("name must be a string or a list of strings.")
    #     if isinstance(name, list) and not all(isinstance(i, str) for i in name):
    #         raise TypeError("All elements in the list must be strings.")

    #     rt = self.get_retention_time()
    #     if isinstance(name, list):
    #         for i in name:
    #             plt.figure(figsize=(12, 5))
    #             plt.plot(rt, np.sum(self.chromatograms[i], axis=1))
    #             plt.xlabel('Retention Time')
    #             plt.ylabel('Intensity')
    #             plt.gca().spines['right'].set_visible(False)
    #             plt.gca().spines['top'].set_visible(False)
    #             plt.show()
    #     elif isinstance(name, str):
    #         plt.figure(figsize=(12, 5))
    #         plt.plot(rt, np.sum(self.chromatograms[name], axis=1))
    #         plt.xlabel('Retention Time')
    #         plt.ylabel('Intensity')
    #         plt.gca().spines['right'].set_visible(False)
    #         plt.gca().spines['top'].set_visible(False)
    #         plt.show()
    
    # def parse_msp_alignment_compounds(self, msp_file_path):
    #     if not isinstance(msp_file_path, str):
    #         raise TypeError("msp_file_path must be a string.")
    #     if not os.path.isfile(msp_file_path):
    #         raise ValueError("msp_file_path must be a valid file path.")

    #     self.alignment_compound_list = {}
    #     with open(msp_file_path, 'r') as file:
    #         spectra = []
    #         name = None
    #         for line in file:
    #             line = line.strip()
    #             if not line and name:
    #                 self.alignment_compound_list[name] = spectra
    #                 spectra = []
    #             elif line.startswith("Name:"):
    #                 name = line.split(": ", 1)[1]
    #             elif line.startswith("Num Peaks:"):
    #                 spectra = []
    #             elif name and spectra is not None:
    #                 mz_int_pair = [float(x) for x in line.split() if x]
    #                 spectra.append(mz_int_pair)

    #     mz_compressed = np.arange(20, 401, 1)

    #     for name, spectra in self.alignment_compound_list.items():
    #         data_dict = {int(mz): intensity for mz, intensity in spectra}
    #         intensity_list = [data_dict.get(mz, 0) for mz in mz_compressed]
    #         self.alignment_compound_list[name] = np.array(intensity_list)

    # def get_alignment_compound_list(self):
    #     return self.alignment_compound_list

    # def normalize_chromatogram(self, chromatogram):
    #     if not isinstance(chromatogram, np.ndarray):
    #         raise TypeError("chromatogram must be a numpy array.")
    #     if chromatogram.ndim != 2:
    #         raise ValueError("chromatogram must be a 2D numpy array.")

    #     norm_factor = np.sum(chromatogram, axis=1)
    #     norm_factor[norm_factor == 0] = 1  # Avoid division by zero
    #     return chromatogram / norm_factor[:, None]
    
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
    
    # def perform_pca(self,chromatograms, n_components=10):
    #     """
    #     Führt eine Principal Component Analysis (PCA) auf einer Matrix von Gaschromatogrammen durch.

    #     Parameters:
    #     chromatograms (np.array): Die Eingabematrix von Gaschromatogrammen.
    #     n_components (int): Die Anzahl der Hauptkomponenten, die extrahiert werden sollen.

    #     Returns:
    #     Tuple: Tuple mit den folgenden Werten:
    #         - pca (PCA-Objekt): Das PCA-Objekt, das die Hauptkomponenten enthält.
    #         - scores (np.array): Die Scores der Daten in den neuen Hauptkomponenten.
    #         - loadings (np.array): Die Loadings (Ladungen) der Variablen.
    #         - explained_variance_ratio (np.array): Der Anteil der Varianz, die durch jede Hauptkomponente erklärt wird.
    #     """
    #     chromatograms = np.array([chromatograms[key] for key in chromatograms.keys()])
    #     # Standardisierung der Daten
    #     scaler = StandardScaler()
    #     chromatograms_std = scaler.fit_transform(chromatograms)

    #     # PCA durchführen
    #     pca = PCA(n_components=n_components)
    #     scores = pca.fit_transform(chromatograms_std)

    #     # Ladungen (Loadings) sind die Koeffizienten der linearen Kombinationen der ursprünglichen Variablen
    #     loadings = pca.components_.T

    #     # Anteil der erklärten Varianz
    #     explained_variance_ratio = pca.explained_variance_ratio_

    #     return pca, scores, loadings, explained_variance_ratio

    # def PCA(self, Chromatograms):
    #     Chromatograms = np.array([Chromatograms[key] for key in Chromatograms.keys()])
    #     Chromatograms = Chromatograms.T
    #     #Chromatograms = Chromatograms - np.mean(Chromatograms, axis = 0)
    #     covariance_matrix = np.cov(Chromatograms.T)
    #     eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    #     idx = np.argsort(eigenvalues)[::-1]
    #     eigenvalues = eigenvalues[idx]
    #     eigenvectors = eigenvectors[:, idx]
    #     # Calculate Loadings and Scores
    #     scores = np.dot(Chromatograms, eigenvectors)
    #     loadings = np.dot(scores, eigenvectors.T)
    #     return scores, loadings, eigenvalues, eigenvectors
    
    # def write_array_to_mzml(self, array, output_file):
    #     # Check if folder "Warped_mzml" exists, if not create it
    #     # Create an MSExperiment object
    #     exp = oms.MSExperiment()

    #     # Define the m/z values (y-axis) and RT values (x-axis)
    #     mz_values = self.mZ_totlist
    #     rt_values = self.get_retention_time()

    #     # Iterate over the RT values (x-axis)
    #     for rt_index, rt in enumerate(rt_values):
    #         # Create a new MSSpectrum object for each RT
    #         spectrum = oms.MSSpectrum()
    #         spectrum.setRT(rt)

    #         # Iterate over the m/z values (y-axis)
    #         for mz_index, mz in enumerate(mz_values):
    #             intensity = array[rt_index, mz_index]
    #             if intensity > 0:  # Only add peaks with non-zero intensity
    #                 peak = oms.Peak1D()
    #                 peak.setMZ(mz)
    #                 peak.setIntensity(intensity)
    #                 spectrum.push_back(peak)

    #         # Add the spectrum to the experiment
    #         exp.addSpectrum(spectrum)

    #     # Write the experiment to an mzML file
    #     oms.MzMLFile().store(output_file, exp)
    
