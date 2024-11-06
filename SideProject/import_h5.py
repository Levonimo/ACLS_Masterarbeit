# Import the h5 file with dask
import dask.array as da
import h5py
import numpy as np
import os
import time as t

# Define path to the h5 file
PATH = 'F:/Documents/MasterArbeit/ACLS_Masterarbeit/SideProject/Data/'

t1 = t.time()
# Load the h5 file
with h5py.File(os.path.join(PATH, 'daten.h5'), 'r') as f:
    # Get the keys of the groups
    keys = list(f.keys())
    # Get the first group
    first_group = f[keys[0]]
    # Get the keys of the datasets
    dataset_keys = list(first_group.keys())
    # Get the first dataset
    first_dataset = first_group[dataset_keys[0]]
    # Get the shape of the first dataset
    shape = first_dataset.shape
    # Get the dtype of the first dataset
    dtype = first_dataset.dtype
    # Create a dask array
    dask_array = da.from_array(np.array(first_dataset))
    # Print the shape and dtype of the dask array
    print(f"Shape: {dask_array.shape}, dtype: {dask_array.dtype}")
    # Print the first 5 elements of the dask array
    print(dask_array[:5].compute())

t2 = t.time()

print(f"Time: {t2-t1} s")