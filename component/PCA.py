# Description: Perform PCA on the data
#
# Load libraries
from sklearn.decomposition import PCA
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
def perform_pca(data, n_components, scaler_name, method_name):
    # check if data is not empty
    if data is None:
        return None

    # check if data is dictionary
    if isinstance(data, dict):
        # make a list of lists from the dictionary
        if scaler_name == 'None':
            data_points = list(data.values())
            flat_points = []
            original_shapes = []
            for i in range(len(data_points)):
                original_shapes.append(data_points[i].shape)
                flat_points.append(data_points[i].flatten())
            data_points = flat_points
        else:
            # if scaler not none, import each value and sum it up in axis 1
            data_points = [np.sum(list(data[key]), axis=1) for key in data.keys()]
            scaler = get_scaler(scaler_name)
            data_points = scaler.fit_transform(data_points)
        
        points_names = list(data.keys())

    # Perform PCA
    pca = PCA(n_components=n_components, svd_solver=get_method(method_name))
    pca.fit(data_points)

    # Loadings
    loadings = pca.components_

    if scaler_name == 'None':
        # reshape loadings to the original shape
        loadings = [loadings[i].reshape(original_shapes[i]) for i in range(len(data_points))]

    scores = pca.transform(data_points)

    # Explained variance
    explained_variance = pca.explained_variance_ratio_

    return scores, loadings, explained_variance