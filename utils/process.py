
#Libraries
import numpy as np
import matplotlib.pyplot as plt

import scipy
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from scipy import signal
from scipy.fft import fftshift

from torch import from_numpy
# from tensorflow.keras.utils import to_categorical
from sklearn.decomposition import PCA

from IPython.display import HTML


data_path = 'D:/programming/Work/ECoG_decoding_moves/data/'

def shapes(*mas):
    shapes = []
    for arr in mas:
        shapes.append(list(arr.shape))
    return shapes

def download_data(which_data, root_data = data_path ):
    if which_data == 'new':
        path  = root_data + 'DataBase_ECOG/fingerflex/data/'
        #choose appropriate patient
        patient_id = 'bp'
        patient_flex = patient_id + '/' + patient_id + '_fingerflex.mat'
        patient_stim = patient_id + '/' + patient_id + '_stim.mat'

        data = scipy.io.loadmat(path + patient_flex)
        data_target = scipy.io.loadmat(path + patient_stim)
        x = data['data']
        y = data['flex']
        target = data_target['stim']
        print('Data from ', which_data, ' dataset',
              '\nSize of out data before preprocessing:', shapes(x, y, target))
    elif which_data == 'old':
        path = root_data + 'BCI_comp_data/'
        patient_id = 'sub1'
        data = scipy.io.loadmat(path + patient_id + '_comp.mat')

        x = data['train_data']
        y = data['train_dg']
        print('Data from ', which_data, ' dataset'
              '\nSize of our data before preprocessing:', shapes(x, y))

    return x, y
def remove_electrodes_bounds(x, y , n_electrodes = 2, bounds = 1000):
    stds = x.std(axis = 0)
    bad_channels = np.argsort(stds)[-n_electrodes:]
    x_del = np.delete(x, bad_channels, 1)
    xtmp = x_del[bounds:-bounds]
    ytmp = y[bounds:-bounds]
    return xtmp, ytmp


def make_boxcox(y):
    """y.shape: [time, fingers]
    """
    y_process = np.zeros_like(y)
    lambs = []
    for i in range(y.shape[1]):
        y_process[:, i], lamb = boxcox(y[:, i])
        lambs.append(lamb)
    return y_process, lambs
def make_inv_boxcox(y, lambs):
    """y.shape: [time, fingers]
    lamb = len(lambs) = fingers
    """
    y_inv = np.zeros_like(y)
    for i in range(y.shape[1]):
        y_inv[:, i] = inv_boxcox(y[:, i], lambs[i])
    return y_inv - ADD_VALUE

#bank filters
def take_coef(fs, n_coefs, band_pass):
    """It uses for bank filters"""
    nyq = 0.5 * fs
    band_pass = np.array(band_pass)
    b = np.zeros([band_pass.shape[0], n_coefs])
    band_pass = band_pass / nyq
    for i, bands in enumerate(band_pass):
        b[i,:] = signal.firwin(n_coefs, bands, pass_zero='bandpass')
    return b
def make_filter(data, filters):
    """
    Output: [time, channels, features]
    """
    data_out = np.zeros([*data.shape, filters.shape[0]])
    for i, channel in enumerate(data.T):
        for j, b in enumerate(filters):
            data_out[:, i,j] =  signal.lfilter(b, [1.0], channel)
    return data_out

#------------------------------------------------------------------------------#

### MAJOR FUNCTIONS
def preprocess_data(ecog_data, fingers_data, DELAY ,
                    make_bank_filter = False, n_coefs = 0, band_freqs = [],
                    make_fft = False, WINDOW_SIZE = 0 ,
                    SHIFT = 0, fs = 1000,
                    make_box_transform = False, make_out_smooth = 0, N_smooth = 1):
    """ecog_data [ms, n_channels]
       fingers_data [ms, n_fingers]
       """
    # reduce lag between signals
    if DELAY == 0:
        x = ecog_data ; y = fingers_data
    else:
        x = ecog_data[:-DELAY, :] ; y = fingers_data[ DELAY:, :]
    # normalize data
    x = (x - x.mean(axis=0))/x.std(axis=0)

    if make_box_transform:
        y, lambs =  make_boxcox(y)
    else:
        y = y - y.mean(axis=0)
        y = y/(y.std(axis=0))
        if make_out_smooth:
            for i in range(y.shape[-1]):
                y[:,i] = np.convolve(y[:,i], np.ones((N_smooth,))/N_smooth, mode='same')
    if make_fft:
        freq, time, x = signal.spectrogram(x.T, fs = fs, nperseg=WINDOW_SIZE,
                                              noverlap= WINDOW_SIZE- SHIFT)
        x = x.transpose([2, 0, 1]) #we want that time segments will be axis = 0
        x = np.log(x+0.0001)
        y = y[WINDOW_SIZE::SHIFT, :]
    if make_bank_filter:
        coefs = take_coef(fs, n_coefs, band_freqs)  # [3, 256]
        x = make_filter(x, coefs)
    #output
    if make_box_transform:
        return x, y, lambs
    return x, y



# reduce number of channels
#------------------------------------------------------------------------------#
"""PCA"""

def fit_pca(X, n_component):
    """X - must be the train data ( not test)"""
    pca = PCA(n_components = n_component)
    pca.fit(X)
    print(len(pca.explained_variance_), np.sum(pca.explained_variance_ratio_))
    return pca
def reduce_dimension(data, reduce):
    return torch.from_numpy(reduce(data)).float()

def make_pca(tensors, n_components):
    """return X with n_components and tensor datatypes
    also use [0][0] because X and Y tensors on i
    """
    pca = fit_pca(tensors[0], n_components)
    mas = [reduce_dimension(tensor, pca) for tensor in tensors]
    return mas

def make_np_pca(np_X, test_size, n_components):
    """return X with n_components and tensor datatypes
    also use [0][0] because X and Y tensors on i
    """
    pca = PCA(n_components = n_components)
    pca.fit(np_X[:test_size])
    mas = pca.transform(np_X)
    return mas
