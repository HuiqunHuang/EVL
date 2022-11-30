import h5py
import torch
import numpy as np
import pandas as pd

def return_threshold(data_file_name, extreme_high_valid_percent, extreme_low_valid_percent):
    f = h5py.File(data_file_name, 'r')
    data = f['data'].value
    f.close()
    valid_data = []
    for i in range(0, len(data)):
        for j in range(0, len(data[i])):
            if int(data[i][j]) != 0:
                valid_data.append(data[i][j])
    valid_data = np.sort(np.array(valid_data))
    index = int(len(valid_data) * extreme_high_valid_percent) - 1
    extreme_high_threshold = valid_data[index]

    index = int(len(valid_data) * extreme_low_valid_percent) - 1
    extreme_low_threshold = valid_data[index]

    return extreme_high_threshold, extreme_low_threshold

def loadDataDate(data_file_name):
    f = h5py.File(data_file_name, 'r')
    data = f['data'].value
    timeinfo = f['date'].value
    f.close()

    return data, timeinfo