import master_function as mf
import Archiv.master_class_alt as mc
import numpy as np
import os
import matplotlib.pyplot as plt

#PATH = "F:/Documents/MasterArbeit/Data"
PATH = "D:/OneDrive - ZHAW/Masterarbeit/Data"

Data = mc.Data_Preparation(PATH)
Data.convert_d_to_mzml()

#Data.interact_with_msdial("F:/ProgramFiles/MSDIAL/MsdialConsoleApp.exe", "GCMS")
#Data.convert_msdial_to_csv()

data_files = Data.get_name_mzml_files()

# Safe the Dictionary Chromatograms to a npy file if the file is not already existing
if not os.path.exists(PATH+'/Chromatograms.npy'):
    Chromatograms = Data.get_list_of_chromatograms(data_files)
    np.save(PATH+'/Chromatograms.npy', Chromatograms)
else:
    Chromatograms = Data.get_list_of_chromatograms(PATH+'/Chromatograms.npy', source_type = 'FromNPY')

print('#############################################################################################################')
'''
# import ms search library with msp file
Data.parse_msp_alignment_compounds('F:/Downloads/Tenax_Decomposition.msp')

Data.compression_of_spectra()

rt = Data.get_retention_time()
list_retentiontime = []
for i in range(len(data_files)):
    indecs, matrix, pre_match = Data.get_rt_of_alignment_compounds(data_files[i])
    list_retentiontime.append(rt[indecs].tolist())
print(list_retentiontime)

print('#############################################################################################################')

data_transposed = list(zip(*list_retentiontime))

# Erstelle die Boxplots
plt.boxplot(data_transposed)
# replace x-axis labels with compound names
plt.xticks([1, 2, 3], ['Nonanal', 'Acetophenone', 'Benzoic acid'])

plt.ylabel('Retention Time')
plt.show()

print('True RT : Nonanal 13.04, Acetophenone 16.99 , Benzoic acid 26.4')

print('#############################################################################################################')

# Make line plot of matrix with retention time
plt.figure(figsize=(10, 8))
# three subplots under each other
plt.subplot(3, 1, 1)
plt.plot(rt, np.sum(Data.get_chromatogram(data_files[-1]), axis = 1))#/(7*10**4))
plt.ylabel('Intensity')
plt.subplot(3, 1, 2)
plt.plot(rt, pre_match[1])
plt.ylabel('Similarity Score')
plt.subplot(3, 1, 3)
plt.plot(rt, matrix[1])
plt.xlabel('Retention Time')
plt.ylabel('Similarity Score')

plt.show()

print('#############################################################################################################')


#Data.plot_chromatogram(data_files[7])
'''
print('#############################################################################################################')
'''
Data.set_comparison_chromatogram(data_files[-1])
Data.aligned_chromatograms(data_files[4])
'''