from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
import numpy as np

def data_scalling(X, NEW_X, scal_x_all):
    scalers = {}
    for i in range(scal_x_all.shape[2]):
        scalers[i] = MinMaxScaler(feature_range=(0, 1))
        scalers[i].fit_transform(scal_x_all[:, :, i])
        scal_x_all[:, :, i] = scalers[i].transform(scal_x_all[:, :, i])
        if i == 0:
            x = scalers[i].transform(x)
            new_x = scalers[i].transform(new_x)
            for t in range(6):
                X[t][:, :, i] = scalers[i].transform(X[t][:, :, i])
                NEW_X[t][:, :, i] = scalers[i].transform(NEW_X[t][:, :, i])
        else:
            for t in range(6):
                if t == 0:
                    X[t][:, :, i] = scalers[i].transform(X[t][:, :, i])
                    NEW_X[t][:, :, i] = scalers[i].transform(NEW_X[t][:, :, i])
                else:
                    if t < i:
                        X[t][:, :, i - 1] = scalers[i].transform(X[t][:, :, i - 1])
                        NEW_X[t][:, :, i - 1] = scalers[i].transform(NEW_X[t][:, :, i - 1])
                    if t > i:
                        X[t][:, :, i] = scalers[i].transform(X[t][:, :, i])
                        NEW_X[t][:, :, i] = scalers[i].transform(NEW_X[t][:, :, i])

def additional_pca_new_scalling(scal_y, y, X_data, new_x_dict, scal_x_dict):
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    scaler_y.fit_transform(scal_y)
    y = scaler_y.transform(y)

    scal_x_all_pca = scal_x_dict.get('scal_x_all')[:, :, 1:5]
    scal_x_all_pca.shape = (scal_x_all_pca.shape[0] * scal_x_all_pca.shape[1], 4)
    pca = PCA(n_components=1)
    scal_XPCAreduced = pca.fit_transform(scal_x_all_pca)

    x_all_pca = np.zeros((len(X_data.get('x_all'), 10, 3)))
    for i in range(len(X_data.get('x_all'))):
        z = np.zeros((10, 3))
        for j in range(len(z)):
            z[j][0] = X_data.get('x_all')[i, j, 0]
            z[j][1] = pca.transform(X_data.get('x_all')[i, j, 1:5].reshape(1, -1))
            z[j][2] = X_data.get('x_all')[i, j, 5]
        x_all_pca[i] = z

    x_6 = np.zeros((len(x_all_pca), 10, 2))
    for i in range(len(x_all_pca)):
        for j in range(10):
            x_6[i][j][0] = x_all_pca[i, j, 0]
            x_6[i][j][1] = x_all_pca[i, j, 2]

    x_pca = x_all_pca[:, :, :2]

    new_x_all_pca = np.zeros((len(new_x_dict.get('new_x_all')), 10, 3))
    for i in range(len(new_x_dict.get('new_x_all'))):
        z = np.zeros((10, 3))
        for j in range(len(z)):
            z[j][0] = new_x_dict.get('new_x_all')[i, j, 0]
            z[j][1] = pca.transform(new_x_dict.get('new_x_all')[i, j, 1:5].reshape(1, -1))
            z[j][2] = new_x_dict.get('new_x_all')[i, j, 5]
        new_x_all_pca[i] = z

    new_x_6 = np.zeros((len(new_x_all_pca), 10, 2))
    for i in range(len(new_x_all_pca)):
        for j in range(10):
            new_x_6[i][j][0] = new_x_all_pca[i, j, 0]
            new_x_6[i][j][1] = new_x_all_pca[i, j, 2]

    new_x_pca = new_x_all_pca[:, :, :2]
    return x_pca, new_x_pca, y