import master_function as mf
import master_class as mc
import numpy as np
import os
import matplotlib.pyplot as plt

Data = mc.Data_Preparation("F:/Documents/MasterArbeit/Data")
Data.convert_d_to_mzml()

#Data.interact_with_msdial("F:/ProgramFiles/MSDIAL/MsdialConsoleApp.exe", "GCMS")
#Data.convert_msdial_to_csv()

data_files = Data.get_name_mzml_files()

# Safe the Dictionary Chromatograms to a npy file if the file is not already existing
if not os.path.exists('F:/Documents/MasterArbeit/Data/Chromatograms.npy'):
    Chromatograms = Data.get_list_of_chromatograms(data_files)
    np.save('F:/Documents/MasterArbeit/Data/Chromatograms.npy', Chromatograms)
else:
    Chromatograms = Data.get_list_of_chromatograms('F:/Documents/MasterArbeit/Data/Chromatograms.npy', Type = 'FromNPY')

print('#############################################################################################################')

# import ms search library with msp file
Data.parse_msp_alignment_compounds('F:/Downloads/Tenax_Decomposition.msp')

Data.compression_of_spectra()
rt = Data.get_retention_time()
list_retentiontime = []
for i in range(len(data_files)):
    indecs, matrix = Data.get_rt_of_alignment_compounds(data_files[i])
    list_retentiontime.append(rt[indecs].tolist())
print(list_retentiontime)

print('#############################################################################################################')

data_transposed = list(zip(*list_retentiontime))

# Erstelle die Boxplots
plt.boxplot(data_transposed)
plt.xlabel('Spalten')
plt.ylabel('Werte')
plt.title('Boxplot für jede der 6 Spalten')
plt.show()

print('#############################################################################################################')

# Make line plot of matrix with retention time
plt.plot(rt, matrix)
plt.plot(rt, np.sum(Data.get_chromatogram(data_files[-1]), axis = 1)/(7*10**4))
plt.xlabel('Retention Time')
plt.ylabel('Similarity Score')

plt.title('Line Plot of Matrix with Retention Time')
plt.show()