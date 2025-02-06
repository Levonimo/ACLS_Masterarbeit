# Description: Perform PCA on the data
#
# Load libraries
from sklearn.decomposition import PCA
#from sklearn.decomposition import SparsePCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
import numpy as np

# =========================================================================================================
# Scaler methods
def get_scaler(scaler_name):
    if scaler_name == 'None':
        return None
    elif scaler_name == 'StandardScaler':
        return StandardScaler()
    elif scaler_name == 'MinMaxScaler':
        return MinMaxScaler()
    elif scaler_name == 'MaxAbsScaler':
        return MaxAbsScaler()
    elif scaler_name == 'RobustScaler':
        return RobustScaler()
    else:
        return None
    
# =========================================================================================================
# get the methods
def get_method(method_name):
    if method_name == 'svd':
        return 'full'
    elif method_name == 'eigen':
        return 'arpack'
    else:
        return None



# =========================================================================================================
# Perform PCA on the data
def perform_pca(data: dict, n_components: int, scaler_name: str, method_name: str, chrom_dim: str) -> tuple:
    # check if data is not empty
    if data is None:
        return None
    
    # check if data is dictionary
    if isinstance(data, dict):
        # make a list of lists from the dictionary
        if chrom_dim == '3D':
            data_points = list(data.values())
            flat_points = []
            original_shapes = np.array(data_points[0]).shape
            for i in range(len(data_points)):
                flat_points.append(data_points[i].flatten())
            data_points = flat_points

        elif chrom_dim == '2D':
            # if scaler not none, import each value and sum it up in axis 1
            data_points = [np.sum(list(data[key]), axis=1) for key in data.keys()]
            
        if scaler_name != 'None':
            scaler = get_scaler(scaler_name)
            data_points = scaler.fit_transform(data_points) 
    
    # Perform PCA
    pca = PCA(n_components=n_components, svd_solver=get_method(method_name))
    # pca = SparsePCA(n_components=n_components)

    pca.fit(data_points)

    # Loadings
    loadings = pca.components_
    

    
    if chrom_dim == '3D':
        loadings_dict = {}
        for i in range(len(loadings)):
            loadings_dict[i] = loadings[i].reshape(original_shapes)
        # reshape loadings to the original shape
        loadings = loadings_dict
        

    scores = pca.transform(data_points)

    # Explained variance
    explained_variance = pca.explained_variance_ratio_
    #explained_variance = [0.5,0.2,0.1,0.05,0.03]

    return scores, loadings, explained_variance