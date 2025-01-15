import pyopenms as oms


file_path = 'R:/agilent/GC-MS/Emma.net/wilv/Data/Daten_MA/mzml/001_A1_1_OOO.mzML'
print(file_path)
exp = oms.MSExperiment()
oms.MzMLFile().load(file_path, exp)

