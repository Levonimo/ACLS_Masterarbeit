import master_function as mf
import master_class as mc

Data = mc.Data_Preparation("F:/Documents/MasterArbeit/Data/")
Data.convert_d_to_mzml()
Data.interact_with_msdial("F:/ProgramFiles/MSDIAL/MsdialConsoleApp.exe", "GCMS")

Data.convert_msdial_to_csv()

Data.get_list_of_chromatograms(['022_A1_1_OOO','012')


# import ms search library with msp file
Data.parse_msp_file('F:/Downloads/Tenax_Decomposition.msp')