from FileOperation.LoadH5 import loadDataDate
import numpy as np
import pandas as pd
import os
from config import Config
DATAPATH = Config().DATAPATH
from numpy import inf
import scipy.stats as ss
import openturns as ot
ot.Log.Show(ot.Log.NONE)
ot.RandomGenerator.SetSeed(0)

def loadDataForModel(datatype, data_file_name, T=24, len_closeness=4, len_day=3, len_distribution=24, len_trend=6, len_test=None, len_val=None, extreme_high=None):
    assert (len_closeness > 0)

    if datatype == "chicago bike 2021":
        cluster_based_bike_data, time_data = loadDataDate(os.path.join(DATAPATH, 'Data\\Chicago\\Bike\\', data_file_name))

    if datatype == "chicago taxi 2021" or datatype == "chicago bike 2021":
        # using the check-in data from 2021-03-10 to 2021-06-10, normal
        # cluster_based_bike_data = cluster_based_bike_data[T * 68:T * 161]
        # using the check-in data from 2021-07-31 to 2021-10-31, heavy rain and strong wind
        # cluster_based_bike_data = cluster_based_bike_data[T * 211:T * 304]
        # using the check-in data from 2021-08-31 to 2021-11-30, Thanks Giving Day
        cluster_based_bike_data = cluster_based_bike_data[T * 242:T * 336]


    extreme_label = []
    for i in range(0, len(cluster_based_bike_data)):
        label = []
        for j in range(0, len(cluster_based_bike_data[i])):
            if int(cluster_based_bike_data[i][j]) >= extreme_high:
                label.append(1)
            else:
                label.append(0)
        extreme_label.append(label)
    extreme_label = np.array(extreme_label)
    print("cluster_based_bike_data shape: " + str(cluster_based_bike_data.shape))
    print("time_data shape: " + str(time_data.shape))
    print("extreme_label shape: " + str(extreme_label.shape))

    label_X, label_Y = [], []
    target_label_Y = []
    extreme_data_X, extreme_data_Y = [], []
    start_index = max([len_closeness, len_distribution, T * (len_day + 2), 7 * T * len_trend,
                       int(len_closeness + 2 * ((len_day / 2) * T * 7 + (1 - len_day % 2) * T - 1))])
    print("Start index: " + str(start_index))
    window = T * len_day
    near_category_X, near_category_y = [], []

    for i in range(start_index, len(cluster_based_bike_data)):
        near_category_X.append(np.asarray([np.vstack([cluster_based_bike_data[j] for j in range(i - len_closeness, i)])]))
        near_category_y.append(np.asarray(cluster_based_bike_data[i]))

        target_label_Y.append(np.asarray([extreme_label[i]]))

        # if dynamic extreme label
        x, y, label_x, label_y = [], [], [], []
        valid_count = 0
        for j in range(i, 0, -T):
            if valid_count < len_day:
                x.append(cluster_based_bike_data[j - 2 * T:j - T])
                y.append(cluster_based_bike_data[j - T])

                v = cluster_based_bike_data[j - 2 * T:j - T + 1]
                mean_value = np.mean(v)
                std_value = np.std(v)
                extreme_high_value = mean_value + std_value
                current_label = []
                for k in range(0, len(v)):
                    cl = []
                    for m in range(0, len(v[k])):
                        if v[k][m] >= extreme_high_value:
                            cl.append(1)
                        else:
                            cl.append(0)
                    current_label.append(cl)
                current_label = np.array(current_label)
                label_x.append(current_label[:-1])
                label_y.append(current_label[-1:][0])
                valid_count += 1
            if valid_count >= len_day:
                break
        if valid_count < len_day:
            print("Not enough data, please adjust start_index " + str(i))

        extreme_data_X.append([x])
        extreme_data_Y.append([y])
        label_X.append([label_x])
        label_Y.append([label_y])

    label_X = np.vstack(label_X)
    label_Y = np.vstack(label_Y)
    target_label_Y = np.vstack(target_label_Y)
    extreme_data_X = np.vstack(extreme_data_X)
    extreme_data_Y = np.vstack(extreme_data_Y)
    near_category_X = np.vstack(near_category_X)
    near_category_y = np.vstack(near_category_y)


    print("near_category_X shape: " + str(near_category_X.shape))
    print("near_category_y shape: " + str(near_category_y.shape))
    print("label_X shape: " + str(label_X.shape))
    print("label_Y shape: " + str(label_Y.shape))
    print("extreme_data_X shape: " + str(extreme_data_X.shape))
    print("extreme_data_Y shape: " + str(extreme_data_Y.shape))
    print("target_label_Y shape: " + str(target_label_Y.shape))

    near_category_X_train, near_category_X_val, near_category_X_test = \
        near_category_X[:-(len_test + len_val)], near_category_X[-(len_test + len_val):-len_test], near_category_X[
                                                                                                   -len_test:]
    near_category_y_train, near_category_y_val, near_category_y_test = \
        near_category_y[:-(len_test + len_val)], near_category_y[-(len_test + len_val):-len_test], near_category_y[
                                                                                                   -len_test:]

    label_X_train, label_X_val, label_X_test = label_X[:-(len_test + len_val)], label_X[-(
            len_test + len_val):-len_test], label_X[-len_test:]
    label_Y_train, label_Y_val, label_Y_test = label_Y[:-(len_test + len_val)], label_Y[-(
            len_test + len_val):-len_test], label_Y[-len_test:]
    extreme_data_X_train, extreme_data_X_val, extreme_data_X_test = \
        extreme_data_X[:-(len_test + len_val)], extreme_data_X[-(len_test + len_val):-len_test], extreme_data_X[
                                                                                                 -len_test:]
    extreme_data_Y_train, extreme_data_Y_val, extreme_data_Y_test = \
        extreme_data_Y[:-(len_test + len_val)], extreme_data_Y[-(len_test + len_val):-len_test], extreme_data_Y[
                                                                                                 -len_test:]
    target_label_Y_train, target_label_Y_val, target_label_Y_test = \
        target_label_Y[:-(len_test + len_val)], target_label_Y[-(len_test + len_val):-len_test], target_label_Y[
                                                                                                 -len_test:]

    print("near_category_X_train shape: " + str(near_category_X_train.shape))
    print("near_category_X_val shape: " + str(near_category_X_val.shape))
    print("near_category_X_test shape: " + str(near_category_X_test.shape))
    print("")
    print("near_category_y_train shape: " + str(near_category_y_train.shape))
    print("near_category_y_val shape: " + str(near_category_y_val.shape))
    print("near_category_y_test shape: " + str(near_category_y_test.shape))
    print("")
    print("label_X_train shape: " + str(label_X_train.shape))
    print("label_X_val shape: " + str(label_X_val.shape))
    print("label_X_test shape: " + str(label_X_test.shape))
    print("")
    print("extreme_data_X_train shape: " + str(extreme_data_X_train.shape))
    print("extreme_data_X_val shape: " + str(extreme_data_X_val.shape))
    print("extreme_data_X_test shape: " + str(extreme_data_X_test.shape))
    print("")
    print("label_Y_train shape: " + str(label_Y_train.shape))
    print("label_Y_val shape: " + str(label_Y_val.shape))
    print("label_Y_test shape: " + str(label_Y_test.shape))
    print("")
    print("target_label_Y_train shape: " + str(target_label_Y_train.shape))
    print("target_label_Y_val shape: " + str(target_label_Y_val.shape))
    print("target_label_Y_test shape: " + str(target_label_Y_test.shape))
    print("")

    X_data_train = []
    X_data_val = []
    X_data_test = []
    for l, X_ in zip([len_closeness, len_day, len_day], [near_category_X_train, label_Y_train, extreme_data_X_train]):
        if l > 0:
            X_data_train.append(X_)
    for l, X_ in zip([len_closeness, len_day, len_day], [near_category_X_val, label_Y_val, extreme_data_X_val]):
        if l > 0:
            X_data_val.append(X_)
    for l, X_ in zip([len_closeness, len_day, len_day], [near_category_X_test, label_Y_test, extreme_data_X_test]):
        if l > 0:
            X_data_test.append(X_)
    Y_data_train = []
    Y_data_val = []
    Y_data_test = []
    for l, X_ in zip([1, 1, 1], [near_category_y_train, target_label_Y_train, label_Y_train]):
        if l > 0:
            Y_data_train.append(X_)
    for l, X_ in zip([1, 1, 1], [near_category_y_val, target_label_Y_val, label_Y_val]):
        if l > 0:
            Y_data_val.append(X_)
    for l, X_ in zip([1, 1, 1], [near_category_y_test, target_label_Y_test, label_Y_test]):
        if l > 0:
            Y_data_test.append(X_)

    for _X in X_data_train:
        print(np.array(_X).shape, )
    print()
    for _X in X_data_val:
        print(np.array(_X).shape, )
    print()
    for _X in X_data_test:
        print(np.array(_X).shape, )
    print()

    return X_data_train, X_data_val, X_data_test, Y_data_train, Y_data_val, Y_data_test, \
           np.array(near_category_X_train), np.array(near_category_X_val), np.array(near_category_X_test), \
           np.array(near_category_y_train), np.array(near_category_y_val), np.array(near_category_y_test), \
           extreme_data_X_train, extreme_data_X_val, extreme_data_X_test, \
           extreme_data_Y_train, extreme_data_Y_val, extreme_data_Y_test, \
           label_Y_train, label_Y_val, label_Y_test, \
           np.array(target_label_Y_train), target_label_Y_val, target_label_Y_test