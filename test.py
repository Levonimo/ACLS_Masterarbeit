import master_function as mf
import numpy as np
import os

#######################################################################################
# Checking import_list_of_mzml_files and plot_a_list_of_chromatograms_array
#######################################################################################

PATH = "F:/Documents/MasterArbeit/Data/_mzml/"
#FILE_NAME = ["001_A1_1_OOO.mzML", "008_A1_1_OOO.mzML", "015_A1_1_OOO.mzML", "022_A1_1_OOO.mzML"]
'''
# get the name of all but only the mzML-files in the directory in a list
FILE_NAME = [f for f in os.listdir(PATH) if f.endswith('.mzML')]


# select key value pairs which key contains xxx_xx_x_OOO.mzML
FILE_NAMES_OOO = [f for f in FILE_NAME if "OOO" in f]
# select key value pairs which key contains xxx_xx_x_SOO.mzML
FILE_NAMES_SOO = [f for f in FILE_NAME if "SOO" in f]
# select key value pairs which key contains xxx_xx_x_SGO.mzML
FILE_NAMES_SGO = [f for f in FILE_NAME if "SGO" in f]
# select key value pairs which key contains xxx_xx_x_SOL.mzML
FILE_NAMES_SOL = [f for f in FILE_NAME if "SOL" in f]
# select key value pairs which key contains xxx_xx_x_SGL.mzML
FILE_NAMES_SGL = [f for f in FILE_NAME if "SGL" in f]


chromatograms_OOO, rt = mf.import_list_of_mzml_files(PATH, FILE_NAMES_OOO)
chromatograms_SOO, rt = mf.import_list_of_mzml_files(PATH, FILE_NAMES_SOO)
chromatograms_SGO, rt = mf.import_list_of_mzml_files(PATH, FILE_NAMES_SGO)
chromatograms_SOL, rt = mf.import_list_of_mzml_files(PATH, FILE_NAMES_SOL)
chromatograms_SGL, rt = mf.import_list_of_mzml_files(PATH, FILE_NAMES_SGL)


mf.plot_a_list_of_chromatograms_array(chromatograms_OOO,rt)
mf.plot_a_list_of_chromatograms_array(chromatograms_SOO,rt)
mf.plot_a_list_of_chromatograms_array(chromatograms_SGO,rt)
mf.plot_a_list_of_chromatograms_array(chromatograms_SOL,rt)
mf.plot_a_list_of_chromatograms_array(chromatograms_SGL,rt)

#######################################################################################
# Checking mean_of_chromatograms
#######################################################################################

mean_chromatogram = mf.mean_of_dict(chromatograms_OOO)


# plot the mean chromatogram
mf.plot_a_list_of_chromatograms_array(np.sum(mean_chromatogram,axis=1),rt)

'''

#######################################################################################
# Checking substract mean chromatogram from another chromatogram
#######################################################################################
'''

FILE_NAME_test = "013_A1_6_SGL.mzML"
chromatogram_test, mz, rt_test = mf.mzml_to_array(PATH+FILE_NAME_test)

# substract the mean chromatogram from the chromatogram_test
substracted_chromatogram = chromatogram_test - mean_chromatogram

# plot the substracted chromatogram and the chromatogram_test in one plot
mf.plot_a_list_of_chromatograms_array([substracted_chromatogram,chromatogram_test,mean_chromatogram],rt_test)
'''

#######################################################################################
# Test interact_with_msdial
#######################################################################################

#Chromatogram , mz, rt = mf.mzml_to_array(PATH)
DATA_PATH = 'F:/Documents/MasterArbeit/OOO_test'
MSDIAL_PATH = "F:/ProgramFiles/MSDIAL/MsdialConsoleApp.exe"
#mf.interact_with_msdial(DATA_PATH, MSDIAL_PATH, "GCMS_MSDIAL_param", type="gcms")

#mf.convert_msdial_to_csv(DATA_PATH)



#######################################################################################
# Searching for  peaks in the csv files which appear in each file
#######################################################################################

x=mf.find_common_peaks_in_csv_files(DATA_PATH)

print(x)


#######################################################################################
# 3D Plot of the Chromatogram
#######################################################################################
'''
# 3D Plot of the Chromatogram

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# create a meshgrid for the x and y axis
x, y = np.meshgrid(mz, rt)
z = Chromatogram
# plot the surface
ax.plot_surface(x, y, z,  cmap='viridis', alpha=0.7)
# define limits for the x and y axis and z axis
ax.set_xlim(20, 400)
ax.set_ylim(5, 28)
ax.set_zlim(0, 5e5)
# make the plot bigger
fig.set_size_inches(10, 10)

# plot the total ion current as a 2D line plot in the y and z plane
ax.plot(rt, np.sum(Chromatogram, axis=1)/100,zs=20 , zdir='x', linewidth=1)
# plot the total ion current as a 2D line plot in the x and z plane
ax.plot(mz,np.sum(Chromatogram, axis=0)/300, zs=28 , zdir='y', linewidth=1)

# set the labels for the x, y and z axis
ax.set_xlabel('m/z')
ax.set_ylabel('Retention Time (min)')
ax.set_zlabel('Total Ion Current')


plt.show()
'''