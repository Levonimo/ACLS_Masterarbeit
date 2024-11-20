import os
import subprocess
import numpy as np


class TransformerInport:
    def __init__(self, path):
        if not isinstance(path, str):
            raise TypeError("Path must be a string.")
        if not os.path.isdir(path):
            raise ValueError("Path must be a valid directory.")

        self.path = path
        self.mzml_path = os.path.join(self.path, "mzml")
        os.makedirs(self.mzml_path, exist_ok=True)

        self.convert_d_to_mzml()
        
        
    def convert_d_to_mzml(self):
        for file in os.listdir(self.path):
            if file.endswith(".D"):
                mzml_file = os.path.join(self.mzml_path, file.replace(".D", ".mzML"))
                if not os.path.exists(mzml_file):
                    command = f"msconvert {os.path.join(self.path, file)} -o {self.mzml_path} --mzML --filter \"peakPicking true 1-\""
                    subprocess.run(command, shell=True, check=True)

    def get_file_names(self):
        files = os.listdir(self.mzml_path)
        #remove the .mzML ending
        files = [file.replace('.mzML','') for file in files if file.endswith('.mzML')]
        return files
    


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