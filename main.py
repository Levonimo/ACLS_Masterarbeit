import master_function as mf
import master_class as mc
import numpy as np
import os
import matplotlib.pyplot as plt

#PATH = "F:/Documents/MasterArbeit/Data"
#PATH = "D:/OneDrive - ZHAW/Masterarbeit/Data"
PATH = 'C:/Users/wilv/OneDrive - ZHAW (1)/Masterarbeit/Data'

Data = mc.DataPreparation(PATH)
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

print('#############################################################################################################')

Data.set_comparison_chromatogram(data_files[-1])
Data.aligned_chromatograms(data_files[4])


print('#############################################################################################################')

Warped_chromatograms = dict()

for i in range(len(data_files)-1):
    Warped_chromatograms[data_files[i]] = Data.warping(np.sum(Chromatograms[data_files[-1]], axis = 1), np.sum(Chromatograms[data_files[i]], axis = 1))

Warped_chromatograms[data_files[-1]] = np.sum(Chromatograms[data_files[-1]], axis = 1)

np.save(PATH+'/Warped_chromatograms.npy', Warped_chromatograms)
'''

Warped_chromatograms = np.load(PATH+'/Warped_chromatograms.npy', allow_pickle=True).item()
print('#############################################################################################################')

rt = Data.get_retention_time()
'''
# Plot the Warped Chromatograms
plt.figure(figsize=(12, 5))
for i in range(len(data_files)):
    plt.plot(rt, Warped_chromatograms[data_files[i]])
plt.xlabel('Retention Time')
plt.ylabel('Intensity')
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.show()

print('#############################################################################################################')

# Plot the Original Chromatograms
plt.figure(figsize=(12, 5))
for i in range(len(data_files)):
    plt.plot(rt, np.sum(Chromatograms[data_files[i]], axis = 1))
plt.xlabel('Retention Time')
plt.ylabel('Intensity')
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.show()
'''
print('#############################################################################################################')
pca, scores, loadings, explained_variance_ratio = Data.perform_pca(Warped_chromatograms)
#scores, loadings, eigenvalues, eigenvectors = Data.PCA(Warped_chromatograms)

# Plot the PCA
plt.figure(figsize=(12, 5))
#plt.plot(eigenvalues[:4]/np.sum(eigenvalues), 'o-')
plt.plot(explained_variance_ratio, 'o-')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
plt.show()

print('#############################################################################################################')


'''
# Plot the PC1 - PC5
plt.figure(figsize=(12, 5))
for i in range(5):
    plt.plot(rt, eigenvectors[:,i])
plt.xlabel('Retention Time')
plt.ylabel('Intensity')
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.legend(['PC1', 'PC2', 'PC3', 'PC4', 'PC5'])
plt.show()
'''

print('#############################################################################################################')

# plot the scores of the first two PCs
plt.figure(figsize=(12, 5))
for i in range(len(data_files)):
    if 'OOO' in data_files[i]:
        plt.scatter(scores[i, 0], scores[i, 1], color = 'silver')
    elif 'SOO' in data_files[i]:
        plt.scatter(scores[i, 0], scores[i, 1], color = 'bisque')
    elif 'SOL' in data_files[i]:
        plt.scatter(scores[i, 0], scores[i, 1], color = 'orange')
    elif 'SGO' in data_files[i]:
        plt.scatter(scores[i, 0], scores[i, 1], color = 'lightgreen')
    elif 'SGL' in data_files[i]:
        plt.scatter(scores[i, 0], scores[i, 1], color = 'forestgreen')
    else:
        plt.scatter(scores[i, 0], scores[i, 1], color = 'blue')
    #plt.text(scores[i, 0], scores[i, 1], data_files[i])
#plt.scatter(scores[:, 0], scores[:, 1])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.show()

print('#############################################################################################################')
Chromatograms2 = dict()
for i in range(len(data_files)):
    Chromatograms2[data_files[i]] = np.sum(Chromatograms[data_files[i]], axis = 1)
pca, scores, loadings, explained_variance_ratio = Data.perform_pca(Chromatograms2)


# Plot the PCA
plt.figure(figsize=(12, 5))
#plt.plot(eigenvalues[:4]/np.sum(eigenvalues), 'o-')
plt.plot(explained_variance_ratio, 'o-')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
plt.show()

# plot the scores of the first two PCs
plt.figure(figsize=(12, 5))
for i in range(len(data_files)):
    if 'OOO' in data_files[i]:
        plt.scatter(scores[i, 0], scores[i, 1], color = 'silver')
    elif 'SOO' in data_files[i]:
        plt.scatter(scores[i, 0], scores[i, 1], color = 'bisque')
    elif 'SOL' in data_files[i]:
        plt.scatter(scores[i, 0], scores[i, 1], color = 'orange')
    elif 'SGO' in data_files[i]:
        plt.scatter(scores[i, 0], scores[i, 1], color = 'lightgreen')
    elif 'SGL' in data_files[i]:
        plt.scatter(scores[i, 0], scores[i, 1], color = 'forestgreen')
    else:
        plt.scatter(scores[i, 0], scores[i, 1], color = 'blue')
    #plt.text(scores[i, 0], scores[i, 1], data_files[i])
#plt.scatter(scores[:, 0], scores[:, 1])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.show()