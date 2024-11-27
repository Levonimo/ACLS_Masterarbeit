# Description: Perform PCA on the data
#
# Load libraries
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler

# =========================================================================================================
# Scaler methods
def get_scaler(scaler_name):
    if scaler_name == 'StandardScaler':
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
def perform_pca(data, n_components,scaler_name, method_name):
    # Scale the data
    scaler = get_scaler(scaler_name)
    data = scaler.fit_transform(data)

    # Perform PCA
    pca = PCA(n_components=n_components, svd_solver=get_method(method_name))
    pca.fit(data)
    return pca.transform(data)