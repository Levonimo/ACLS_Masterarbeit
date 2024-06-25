import master_function as mf
import master_class as mc

Data = mc.Data_Preparation("F:/Documents/MasterArbeit/Data")
Data.convert_d_to_mzml()
Data.interact_with_msdial("F:/ProgramFiles/MSDIAL/MsdialConsoleApp.exe", "GCMS")

Data.convert_msdial_to_csv()

data_files = Data.get_name_mzml_files()
print(data_files)

Data.get_list_of_chromatograms(data_files)

Data.get_chromatogram(data_files[0])


print('#############################################################################################################')
# import ms search library with msp file
Data.parse_msp_alignment_compounds('F:/Downloads/Tenax_Decomposition.msp')

Data.compression_of_spectra()

indecs = Data.get_rt_of_alignment_compounds(data_files[5])
print(indecs)
rt = Data.get_retention_time()
print(rt[indecs])