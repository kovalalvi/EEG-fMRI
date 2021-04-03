

import numpy as np
import matplotlib.pyplot as plt

import scipy
import scipy.io
from scipy.stats import boxcox
from scipy import signal
from scipy.special import inv_boxcox
from scipy.interpolate import interp1d


from shutil import copyfile

from IPython.display import HTML

def interpolate_data(data, desire_lenght=305000):
    """ 
    data shape is [n_electrode, time_lenght]
    Returns:
    interp(data) shape is [n_electrode, desire_lenght]
    """
    output_lenght = desire_lenght

    n_roi, time_step = data.shape
    
    # estimate interpolation function 
    x = np.arange(time_step)
    interp_func = interp1d(x, data, kind='cubic', axis = 1)

    # calculate interpolated data
    x_new = np.linspace(0, time_step-1, output_lenght)
    fmri_interp = interp_func(x_new)
    return fmri_interp
    

def shapes(*mas):
    shapes = []
    for arr in mas:
        shapes.append(list(arr.shape))
    return shapes

def show_shapes(*mas):
    shapes = []
    for arr in mas:
        shapes.append(list(arr.shape))
    return shapes


def exp_move_average(data, alpha=0.5):
    N = len(data)
    mas_out = np.zeros(N)
    for  i in range(1,N):
        mas_out[i] = alpha*data[i] + (1-alpha)*mas_out[i-1]
    return mas_out


def display_animation(anim):
    plt.close(anim._fig)
    return HTML(anim.to_jshtml())
