import mne
from preprocessing import eeg_transform, fmri_transform
from nilearn import image
import numpy as np
from fMRI_Regions import get_masked_fmri


# create_dataset

def normalize(data, return_values = False):
    """ 
    data : [n_channels, t_step]
    Calculate mean and std for each cnahnels. Make transforamtion 
    """
    means = np.mean(data, axis = 1, keepdims=True)
    print(means.shape)
    stds = np.std(data, axis = 1, keepdims=True)
    transform_data = (data - means)/stds
    if return_values:
        return transform_data, (means, stds)
    return transform_data

class Dataset:
    def __init__(self, random_seed=42, segment_length=16384, eeg_padded=False,
                 frame_creation_time=1950, step=195):
        self.eeg_padded = eeg_padded
        self.segment_length = segment_length
        self.frame_creation_time = frame_creation_time
        self.step = step # between segments in EEG and fMRI 
        self.random_state = np.random.RandomState(random_seed)
        self.normalize_means_stds = 0

    def create_dataset(self, start_time, end_time, delay, fmri_end, eeg_path, fmri_path):
        vector_exclude = ['EOG', 'ECG', 'CW1', 'CW2', 'CW3', 'CW4', 'CW5', 'CW6', 'Status']
        
        # download fMRI data
        raw = mne.io.read_raw_edf(eeg_path, exclude=vector_exclude)
        eeg_raw = raw.get_data()
        eeg = mne.filter.filter_data(eeg_raw, sfreq=1000, l_freq=5, h_freq=100)
        eeg_flip = np.fliplr(eeg)
        
        # normalize input data
        eeg, self.normalize_means_stds = normalize(eeg, True)
#         eeg_flip, normalize_means_stds = normalize(eeg_flip, True)
        
        # download fMRI data
        fmri_im = image.smooth_img(fmri_path, fwhm=6)
        fmri = get_masked_fmri(fmri_im, "sub")
        
        print('Size of raw EEG data: ', eeg_raw.shape)
        print('Size of FFT of raw EEG data: ', eeg.shape)
        print('Size of fMRI data: ', fmri_im.shape)
        
        
        start = start_time
        end = start_time + self.segment_length
        x_list = []
        y_list = []
        x_fl_list = []
        while end < eeg.shape[1] and end <= fmri_end and end < end_time:
            
            # retrieve spectrogram from some segment
            signal = eeg[..., start:end]
            x = eeg_transform(signal)
            
            signal_flip = eeg_flip[..., start:end]
            x1 = eeg_transform(signal_flip)
            
            y = fmri_transform(end, fmri, delay, fmri_end)
            
            x_list.append(x)
            y_list.append(y)
            x_fl_list.append(x1)
            start += self.step
            end += self.step
        x_list = np.array(x_list)
        x_fl_list = np.array(x_fl_list)
        y_list = np.array(y_list)
        return x_list, y_list, x_fl_list
