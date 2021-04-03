#Libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy import signal
from scipy.fft import fftshift


# from tensorflow.keras.utils import to_categorical
from torch import from_numpy, split
from torch.utils.data import Dataset, TensorDataset, Subset

## - Dataset Class
## - Balance data
## - One hot encode



#------------------------------------------------------------------------------#
"""Dataset Class"""

class CreateDataset(Dataset):
    """
    work with raw(it is unneccesary), many_to_many task
    - Input data with reduce delay between them
    - Tx (int) : size of window
    - stride (int) : which step between window
    
    - freq_border ; (min , max)
    - to_many (bool): seq_len of output.
        if to_many= True -> seq_len = seq_len
        if to_many= True -> seq_len = 1
    - OUTPUT:  size = len(x_tensor)
        - if time_first = True
                X = [size, seq_len, n_channels]
                Y = [size, seq_len, 1]
        - else
                X = [size, n_channels, 1, seq_len]
                Y = [size, seq_len, 1]
    """
    def __init__(self, x_tensor, y_tensor, Tx=60, to_many=True, stride=1, time_first=False,
                 make_fft = False, WINDOW_SIZE = 256, SHIFT = 32, fs = 1000, freq_border=(0, 100), 
                 classify = False,  n_classes = 100, smooth_out= False, std_value = 0):

        self.Tx = Tx  #size of window
        self.stride = stride # which step between window
        self.time_first = time_first
        self.to_many = to_many

        self.x = x_tensor
        self.y = y_tensor
        
        # preprocessing X
        self.make_fft = make_fft
        self.SHIFT = SHIFT
        self.WINDOW_SIZE = WINDOW_SIZE
        self.fs = fs
        self.min_freq = freq_border[0]
        self.max_freq = freq_border[1]
        
        # settings classification task
        self.classify = classify
        self.min_value = y_tensor.min().item()
        self.max_value = y_tensor.max().item()
        self.n_classes = n_classes
        self.smooth_out = smooth_out
        self.std_value = std_value
    def __getitem__(self, idx):
        start_time = idx*self.stride
        end_time = start_time + self.Tx

        if self.make_fft:
            gap= (self.Tx-1)*self.SHIFT + self.WINDOW_SIZE
            # output -> [n_channel, n_features, seq_len(Tx)]
            freq, time, x = signal.spectrogram(self.x[start_time:start_time+gap].T,
                                               fs = self.fs, nperseg=self.WINDOW_SIZE,
                                               noverlap= self.WINDOW_SIZE- self.SHIFT)
            
            
            ### maximum and minimum freq bounding...
            idx_min = int(np.argwhere(freq > self.min_freq)[0])
            idx_max = int(np.argwhere(freq < self.max_freq)[-1])
            freq = freq[idx_min:idx_max]
            x = x[:, idx_min:idx_max, :]
#             print(shapes(freq, time, x))
            
            x = np.log(x+0.0001)
            start_y = self.WINDOW_SIZE+start_time
            end_y = start_time + gap + self.SHIFT
            Y_dataset = self.y[start_y:end_y:self.SHIFT, :]
            if self.time_first:
                X_dataset = x.transpose([2, 0, 1])
            else:
                X_dataset = x
        else:
            if self.time_first:
                X_dataset = self.x[start_time:end_time]
            else:
                X_dataset = self.x[start_time:end_time].T
                # are there any filters
                if len(self.x.shape)==3:
                    X_dataset = X_dataset.permute(1,0, 2)
                else:
                    X_dataset = X_dataset.unsqueeze(dim = -2)
            Y_dataset = self.y[start_time:end_time]
        if self.to_many:
            Y_dataset = Y_dataset
        else:
            Y_dataset = Y_dataset[-1:]  # keep dimensions

        if self.classify:
            Y_dataset = one_hot_encode(Y_dataset, self.n_classes, self.min_value, self.max_value,
                                       self.smooth_out, self.std_value)
        return (X_dataset, Y_dataset)
    def __len__(self):
        size = self.x.shape[0]
        if self.make_fft:
            size = (size-self.WINDOW_SIZE)//self.SHIFT + 1
        return (size-self.Tx)//self.stride
    def __repr__(self):
        info = []
        str1 = "\nPyTorch DataSet of our data. \nContain:\nX: " +str(shapes(self.x)) + '\nY: '+ str(shapes(self.y))
        str_par = '\nParameters:'
        str2 = "\n- MANY_TO_MANY: " + str(self.to_many)
        str3 = "\n- Lenght of time(Tx): " + str(self.Tx)
        str4 = "\n- Step(stride): " + str(self.stride)
        str4_a = "\n- time_first: " + str(self.time_first)
        x, y = self[0]
        str5 = '\nOutput:\nSize of dataset: ' + str(len(self))
        str6 = '\nX: ' + str(shapes(x)) + '\nY: ' +  str(shapes(y))
        return str1 + str_par + str2 + str3 + str4 +str4_a+ str5 + str6

    
    
class CreateDatasetCustom(Dataset):
    """
    work with raw(it is unneccesary), many_to_many task
    - Input data with reduce delay between them
        x_tensor shapes [times, n_channels]
        y_tensor shapes [times, n_channels]
        
    - Tx (int) : size of window
    - stride (int) : which step between window
    
    - freq_border ; (min , max)
    - to_many (bool): seq_len of output.
        if to_many= True -> seq_len = seq_len
        if to_many= True -> seq_len = 1
    ----------------------------------------------------
    - Returns:  size = len(x_tensor)
        - if time_first = True
                X = [size, seq_len, n_channels]  # output -> [n_channel, n_features, seq_len(Tx)]
                Y = [size, seq_len, 1]
        - else
                X = [size, n_channels, 1, seq_len]
                Y = [size, seq_len, 1]
    """
    def __init__(self, x_tensor, y_tensor, Tx=60, to_many=True, stride=1, time_first=False,
                 make_fft = False, WINDOW_SIZE = 256, SHIFT = 32, fs = 1000, freq_border=(0, 100), 
                 classify = False, n_classes = 100, smooth_out= False, std_value = 0):

        self.Tx = Tx  #size of window
        self.stride = stride # which step between window
        self.time_first = time_first
        self.to_many = to_many

        self.x = x_tensor
        self.y = y_tensor
        
        # preprocessing X
        self.make_fft = make_fft
        self.SHIFT = SHIFT
        self.WINDOW_SIZE = WINDOW_SIZE
        self.fs = fs
        self.min_freq = freq_border[0]
        self.max_freq = freq_border[1]
        
        self.gap = (self.Tx-1)*self.SHIFT + self.WINDOW_SIZE #it is advansed WINDOW which we slide on input eeg data
        
        # settings classification task
        self.classify = classify
        self.min_value = y_tensor.min().item()
        self.max_value = y_tensor.max().item()
        self.n_classes = n_classes
        self.smooth_out = smooth_out
        self.std_value = std_value
    def __getitem__(self, idx):
        

        if self.make_fft:
            start_time = idx*self.stride
            end_time = start_time + self.Tx
            
            
            # output -> [n_channel, n_features, seq_len(Tx)]
            freq, time, x = signal.spectrogram(self.x[start_time:start_time+self.gap].T,
                                               fs = self.fs, nperseg=self.WINDOW_SIZE,
                                               noverlap= self.WINDOW_SIZE- self.SHIFT)
            
            
            ### maximum and minimum freq bounding...
            idx_min = int(np.argwhere(freq > self.min_freq)[0])
            idx_max = int(np.argwhere(freq < self.max_freq)[-1])
            freq = freq[idx_min:idx_max]
            x = x[:, idx_min:idx_max, :]
            
            ### some normalization of Spectrogram
            x = np.log(x+0.0001)
            if self.time_first:
                X_dataset = x.transpose([2, 0, 1])
            else:
                X_dataset = x
            
            
            start_y = start_time+self.WINDOW_SIZE
            end_y = start_y + self.gap
            Y_dataset = self.y[start_y:end_y, :]

        else:
            start_time = idx*self.stride
            end_time = start_time + self.Tx
            
            if self.time_first:
                X_dataset = self.x[start_time:end_time]
            else:
                X_dataset = self.x[start_time:end_time].T
#                 # are there any filters
#                 if len(self.x.shape)==3:
#                     X_dataset = X_dataset.permute(1,0,2)
#                 else:
#                     X_dataset = X_dataset.unsqueeze(dim = -2)
            Y_dataset = self.y[start_time:end_time]
        
        
        if self.to_many:
            Y_dataset = Y_dataset
        else:
            Y_dataset = Y_dataset[-1:]  # keep dimensions

        if self.classify:
            Y_dataset = one_hot_encode(Y_dataset, self.n_classes, self.min_value, self.max_value,
                                       self.smooth_out, self.std_value)
        return (X_dataset, Y_dataset)
    def __len__(self):
        input_size = self.x.shape[0]
        if self.make_fft:
            output_size = (input_size - self.gap)//self.stride + 1 
        else:
            output_size = (input_size - self.Tx)//self.stride + 1
        return output_size
    def __repr__(self):
        info = []
        str1 = "\nPyTorch DataSet of our data. \nContain:\nX: " +str(shapes(self.x)) + '\nY: '+ str(shapes(self.y))
        str_par = '\nParameters:'
        str2 = "\n- MANY_TO_MANY: " + str(self.to_many)
        str2a = "\n- Advanced lenght of time for FFT only(gap): " + str(self.gap)
        str3 = "\n- Lenght of time(Tx): " + str(self.Tx)
        str4 = "\n- Step(stride): " + str(self.stride)
        str4_a = "\n- time_first: " + str(self.time_first)
        x, y = self[0]
        str5 = '\nOutput:\nSize of dataset: ' + str(len(self))
        str6 = '\nX: ' + str(shapes(x)) + '\nY: ' +  str(shapes(y))
        return str1 + str_par + str2 +str2a+ str3 + str4 +str4_a+ str5 + str6


def ratio_to_num(ratio_list, size):
    """ratio_list - percentage
       size - size of dataset which you want chunks
    """
    ratio = [0]*len(ratio_list)
    for i in range(len(ratio_list)):
        ratio[i] = ratio_list[i]*size
        ratio[i] = int(ratio[i])
    ratio[-1] = size - sum(ratio[:-1])
    return ratio
def make_split(data, ratioes = [0.7, 0.15, 0.15]):
    """data - torch.tensor
    ratio_list - list of percents in each category
    return: train/val/test tensors without shuffle
    """
    am = ratio_to_num(ratioes, data.shape[0])
    X = split(data, am)
    return X

def datasets_creation(x, y, ratio_list, Tx, to_many, stride, time_first,
                      fft_set, classify_dict):
    x_tensor = from_numpy(x).float()
    y_tensor = from_numpy(y).float()

    x_tensors = make_split(x_tensor, ratioes = ratio_list)
    y_tensors = make_split(y_tensor, ratioes = ratio_list)

    datasets = []
    for x_tensor, y_tensor in zip(x_tensors, y_tensors):
        dataset_tmp = CreateDatasetCustom(x_tensor, y_tensor, Tx, to_many, stride,
                                          time_first, **fft_set, **classify_dict)
        datasets.append(dataset_tmp)
    return datasets


#------------------------------------------------------------------------------#
"""Balance data"""

def get_moves_intervals(data, threshold = 1 , set_bound = 1000, add_rest =0 ):
    """This function return coordinates of moves (begin, end)
        based on threshold
        data = (time, n_finger)
        treshool set power of move
        set_bound how many rest point after moving(how we separate different moves)
    """
    intervals_all = []
    for signal in data.T:
        intervals = []
        begin = 0; end = 0
        signal = np.where(signal > threshold, 1, 0)
        i = -1
        while i < len(signal)-1:
            i += 1
            # find first move
            if signal[i] == 1:
                begin = i
                # try find end of moves
                while i < len(signal) - set_bound:
                    i += 1
                    # we need that no any moves in set_bound ms after
                    # if first bool True than it don't check second
                    if  signal[i] == 0 and  np.all(signal[i:i+set_bound]== 0):
                        end = i
                        #adding additional points before anf after moves
                        begin -= add_rest ; end += add_rest
                        begin = begin if begin >= 0  else 0
                        end = len(signal)-1 if end  >= len(signal) else end
                        intervals.append([begin, end])
                        break
                    else:
                        continue
        #adding for each finger
        intervals_all.append(intervals)
    return intervals_all
def get_moves_indexes(intervals_data):
    """
    Transform intervals into indexes
    Input massive of intervals
    Output massive of indexes(unique)
    """
    #because massive different lenghts
    indexes_whole = []
    for intervals in intervals_data:
        indexes = []
        for interval in intervals:
            indexes_tmp = np.arange(interval[0], interval[1])
            #massive with indexes where there is moving as one list(numpy)
            indexes = np.append(indexes, indexes_tmp)
        # adding only unique values
        indexes_whole.append(np.unique(indexes).astype(int))
    return indexes_whole
def choose_moves(data, indexes):
    """
    input data [time, fingers]
    indexes = [fingers, idx_moves]
    output --> [fingers, moves] different moves
    """
    y_moves = []
    for i, signal in enumerate(data.T):
        y_moves.append(signal[indexes[i]])
    return y_moves

def balanced_dataset(dataset, threshold, how_balanced,
                     n_upsample = 0, n_keep = 100,
                     set_bound= 20, add_rest = 20):
    """
    we keep all sample in which y> threshold
    1) upsample:
        repeating n_upsample times y values > threshold
    2) downsample_random
        take randomly N value from y < threshold
    3) downsample(prefered approach)
        - detect moves( with set bound)
        - add point with add rest( start - add_rest, end + add_rest)
    Input: torch dataset x, y
    Output : torch dataset more balanced
    """
    y = []
    for i in range(len(dataset)):
        # take last element in target y
        y.append(dataset[i][1][-1].item())
    y_value = np.array(y).reshape(-1, 1)


    if how_balanced == 'upsample':
        all_index = np.arange(len(y_value))
        moves_idx = np.where(y_value > threshold)[0]
        upsample_idx = np.repeat(moves_idx, n_upsample)
        keep_list = np.append(all_index, upsample_idx)

    elif how_balanced == 'downsample_random':
        keep_idx = np.where(y_value > threshold)[0]
        remove_list = np.where(y_value <= threshold)[0]
        keep_y = np.random.choice(remove_list, n_keep, replace=False)
        keep_list = np.append(keep_idx,  keep_y)

    elif how_balanced == 'downsample':
        all_intervals = get_moves_intervals(y_value, threshold, set_bound , add_rest)
        keep_list = get_moves_indexes(all_intervals)
        # I use balanced only for one finger.
        keep_list = keep_list[0]
    else:
        print('Choose correctly "how balanced"')
        return False

    dataset_balanced = Subset(dataset, keep_list)
    print('Size before balanced', len(dataset))
    print('Size after balanced', len(dataset_balanced))
    return dataset_balanced


#------------------------------------------------------------------------------#
"""One hot encode"""

def one_hot_encode(data,  n_classes = 100, min_value = 0, max_value = 1,
                   smooth_out = False,  std_value = 0.004):
    """
    Examples: [0, 0.1), [0.1, 0.2)
    input: [n_samples, n_fingers]
    output: [[n_samples, n_fingers, n_classes]
    if smoout: [0,0 ... 0.1, 0.8, 0.1, 0 ... 0]
    if not     [0,0 ...   0,   1,   0, 0 ... 0]
    and intervals [n_classes]
    it is convinient for calculation mathamtical expectation
    """

    intervals_tmp = np.linspace(min_value, max_value, n_classes+1)
    intervals = intervals_tmp[1:-1]
    if smooth_out:
        y_one_hot = np.zeros([data.shape[0], data.shape[1], n_classes])
        for j in range(y_one_hot.shape[1]):
            for sample in range(y_one_hot.shape[0]):
                mas, _ = np.histogram(np.random.normal(data[sample, j],std_value, 500),
                                      bins = n_classes, range=[min_value, max_value])
                pdf = mas/500
                y_one_hot[sample, j] = pdf
    else:
        out = np.digitize(data, intervals, right=False)
        y_one_hot = to_categorical(out, num_classes=n_classes)
        y_one_hot = np.expand_dims(y_one_hot, axis=1) if len(y_one_hot.shape)==2 else y_one_hot
    # output

    y_classes = from_numpy(y_one_hot).float()
    return y_classes

def return_intervals(min_value, max_value, n_classes):
    intervals_tmp = np.linspace(min_value, max_value, n_classes+1)
    intervals = intervals_tmp[:-1]
    return from_numpy(intervals)
def math_expectation(one_hot_encode, intervals):
    """
    one_hot_encode = [n_samples, n_finger, n_classes]
    intervals [n_classes]
    E = x*p(x)
    output: [n_samples, n_fingers] it is value not class
    """
    regression = (one_hot_encode*intervals).sum(axis = -1)
    return regression
def math_variance(pdf, intervals):
    """one_hot_encode = [n_samples, n_finger, n_classes]
    intervals [n_classes]
    """
    expect = math_expectation(pdf, intervals)

    # expect [n_samples, n_finger, 1]
    error = intervals-expect.unsqueeze(-1)
    expect_error =(error**2)*pdf
    return expect_error.sum(-1)


def shapes(*mas):
    shapes = []
    for arr in mas:
        shapes.append(list(arr.shape))
    return shapes