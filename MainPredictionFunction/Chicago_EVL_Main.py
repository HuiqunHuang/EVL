# -*- coding: utf-8 -*-
"""
Usage:
    THEANO_FLAGS="device=gpu0" python exptBikeNYC.py
"""

from __future__ import print_function
import os
import time

import numpy as np
import math
from matplotlib import pyplot
import tensorflow as tf
from tensorflow.python.client.session import Session
from tensorflow.python.framework.ops import get_default_graph

from DataProcessing.PrepareDataForModel import loadDataForModel
from FileOperation.LoadH5 import return_threshold
from Model.Chicago_EVL_Model import MultiTimeSeriesPredictionModel
from config import Config
import metrics as metrics
import random
os.environ['TF_DETERMINISTIC_OPS'] = '1'
SEED = 1337
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

from tensorflow.python.keras.backend import set_session
sess = Session()
graph = get_default_graph()
set_session(sess)

DATAPATH = Config().DATAPATH
nb_epoch_cont = 500
batch_size = 96
T = 24
lr = 0.0002
len_closeness = 2
len_distribution = 1 * T
pdf_ratio = 1000
len_day = 2
len_trend = 2
days_test = 10
days_val = 5
len_test = T * days_test
len_val = T * days_val
category_num = 8
nb_flow, height, width = 1, 4, 3
path_result = 'RET'
path_model = 'MODEL'
if os.path.isdir(path_result) is False:
    os.mkdir(path_result)
if os.path.isdir(path_model) is False:
    os.mkdir(path_model)
delta0 = 1
delta1 = 1000
delta2 = 1000
dropout = 0.2

def build_model():
    model = MultiTimeSeriesPredictionModel(categorynum=category_num, len_closeness=len_closeness, len_day=len_day, T=T)
    return model

@tf.function
def train_step_function(model, x_batch_train, y_batch_train, optimizer):
    with tf.GradientTape() as tape:
        logits = model(x_batch_train, training=True)
        mse = metrics.mean_squared_error(y_batch_train[0], logits[0])
        evl_l1 = metrics.EVL(y_batch_train[1], logits[1])
        gro = tf.reshape(y_batch_train[2], [batch_size * len_day, category_num])
        pre = tf.reshape(logits[2], [batch_size * len_day, category_num])
        evl_l2 = metrics.EVL(gro, pre)
        loss_value = delta0 * mse + delta1 * evl_l1 + delta2 * evl_l2
    grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return mse, evl_l1, evl_l2, loss_value

@tf.function
def val_step_function(x_batch_val, y_batch_val, model):
    val_logits = model(x_batch_val, training=False)
    val_mse = metrics.mean_squared_error(y_batch_val[0], val_logits[0])
    val_evl_l1 = metrics.EVL(y_batch_val[1], val_logits[1])
    gro = tf.reshape(y_batch_val[2], [batch_size * len_day, category_num])
    pre = tf.reshape(val_logits[2], [batch_size * len_day, category_num])
    val_evl_l2 = metrics.EVL(gro, pre)
    val_loss_value = delta0 * val_mse + delta1 * val_evl_l1 + delta2 * val_evl_l2

    return val_mse, val_evl_l1, val_evl_l2, val_loss_value

def main(project_path, datatype, data_file_name, city, extreme_high_valid_percent, extreme_low_valid_percent):
    print("loading data...")
    extreme_high, extreme_low = return_threshold(project_path + "Data\\Chicago\\Bike\\" + data_file_name, extreme_high_valid_percent, extreme_low_valid_percent)
    print("Extreme high: " + str(extreme_high) + ", extreme low: " + str(extreme_low))

    X_data_train, X_data_val, X_data_test, Y_data_train, Y_data_val, Y_data_test, \
    near_category_X_train, near_category_X_val, near_category_X_test, \
    near_category_y_train, near_category_y_val, near_category_y_test, \
    extreme_data_X_train, extreme_data_X_val, extreme_data_X_test, \
    extreme_data_Y_train, extreme_data_Y_val, extreme_data_Y_test, \
    label_Y_train, label_Y_val, label_Y_test, \
    target_label_Y_train, target_label_Y_val, target_label_Y_test = \
        loadDataForModel(datatype, data_file_name, T=T, len_closeness=len_closeness, len_day=len_day,
                         len_distribution=len_distribution, len_trend=len_trend, len_test=len_test,
                         len_val=len_val, extreme_high=extreme_high)

    near_category_X_train = near_category_X_train.astype('float32')
    near_category_y_train = near_category_y_train.astype('float32')
    extreme_data_X_train = extreme_data_X_train.astype('float32')
    label_Y_train = label_Y_train.astype('float32')
    target_label_Y_train = target_label_Y_train.astype('float32')

    train_dataset_X = tf.data.Dataset.from_tensor_slices((near_category_X_train, label_Y_train, extreme_data_X_train))
    train_dataset_Y = tf.data.Dataset.from_tensor_slices((near_category_y_train, target_label_Y_train, label_Y_train))
    train_dataset = tf.data.Dataset.zip((train_dataset_X, train_dataset_Y)).shuffle(buffer_size=1024).batch(batch_size,
                                                                                                            drop_remainder=True)

    near_category_X_val = near_category_X_val.astype('float32')
    near_category_y_val = near_category_y_val.astype('float32')
    extreme_data_X_val = extreme_data_X_val.astype('float32')
    label_Y_val = label_Y_val.astype('float32')
    target_label_Y_val = target_label_Y_val.astype('float32')

    val_dataset_X = tf.data.Dataset.from_tensor_slices((near_category_X_val, label_Y_val, extreme_data_X_val))
    val_dataset_Y = tf.data.Dataset.from_tensor_slices((near_category_y_val, target_label_Y_val, label_Y_val))
    val_dataset = tf.data.Dataset.zip((val_dataset_X, val_dataset_Y)).shuffle(buffer_size=1024).batch(batch_size,
                                                                                                      drop_remainder=True).repeat(100)


    model = build_model()
    optimizer = tf.optimizers.Adam(learning_rate=lr)
    train_mse_all, train_evl_l1_all, train_evl_l2_all, train_loss_all = [], [], [], []
    val_mse_all, val_evl_l1_all, val_evl_l2_all, val_loss_all = [], [], [], []

    best_val = 1000000000
    hyperparams_name = 'c{}.len_day{}.lr{}.batchsize{}'.format(len_closeness, len_day, lr, batch_size)
    fname_param = os.path.join('MODEL', '{}.best.h5'.format(hyperparams_name))

    best_l2 = 1000000000
    for epoch in range(nb_epoch_cont):
        # print("\nStart of epoch %d" % (epoch,))
        start_time = time.time()
        train_total_mse, train_total_evl_l1, train_total_evl_l2, train_total_loss = 0, 0, 0, 0
        val_total_mse, val_total_evl_l1, val_total_evl_l2, val_total_loss = 0, 0, 0, 0

        train_step = 0
        val_step = 0
        train_l2_step = 0
        for x_batch_train, y_batch_train in train_dataset:
            mse, evl_l1, evl_l2, loss_value = train_step_function(model, x_batch_train, y_batch_train, optimizer)
            train_total_mse += mse
            train_total_evl_l1 += evl_l1
            train_total_evl_l2 += evl_l2
            train_total_loss += loss_value
            train_step += 1
            train_l2_step += 1
            if evl_l2 < best_l2:
                best_l2 = evl_l2

        for x_batch_val, y_batch_val in val_dataset:
            val_mse, val_evl_l1, val_evl_l2, val_loss_value = val_step_function(x_batch_val, y_batch_val, model)
            val_total_mse += val_mse
            val_total_evl_l1 += val_evl_l1
            val_total_evl_l2 += val_evl_l2
            val_total_loss += val_loss_value
            val_step += 1


        if val_total_loss / val_step < best_val:
            best_val = val_total_loss / val_step
            model.save_weights(os.path.join('MODEL', '{}.best.h5'.format(hyperparams_name)), overwrite=True)
            model.load_weights(fname_param)

        train_mse_all.append(train_total_mse / train_step)
        train_evl_l1_all.append(train_total_evl_l1 / train_step)
        train_evl_l2_all.append(train_total_evl_l2 / train_l2_step)
        train_loss_all.append(train_total_loss / train_step)
        val_mse_all.append(val_total_mse / val_step)
        val_evl_l1_all.append(val_total_evl_l1 / val_step)
        val_evl_l2_all.append(val_total_evl_l2 / val_step)
        val_loss_all.append(val_total_loss / val_step)
        print(
            '| epoch {:3d} | time: {:5.2f}s | train_mse {:5.4f} | train_evl_l1 {:5.4f} | '
            'train_avg_evl_l2 {:5.4f} | train_loss {:5.4f} | train_best_evl_l2 {:5.4f} | val_mse {:5.4f} | '
            'val_evl_l1 {:5.4f} | val_evl_l2 {:5.4f} | val_loss {:5.4f}'.format(
                epoch, (time.time() - start_time), train_total_mse / train_step, train_total_evl_l1 / train_step,
                train_total_evl_l2 / train_step, train_total_loss / train_step, best_l2,
                val_total_mse / val_step, val_total_evl_l1 / val_step, val_total_evl_l2 / val_step, val_total_loss / val_step))

    total_mae_high = 0
    total_mse_high = 0
    total_msle_high = 0
    total_mape_high = 0
    length_high = 0
    total_ground_truth_high = 0

    total_mae_normal = 0
    total_mse_normal = 0
    total_msle_normal = 0
    total_mape_normal = 0
    length_normal = 0
    total_ground_truth_normal = 0

    label_original, label_predicted = [], []
    y_predict = model.predict(X_data_test)
    p_preicted_data, p_groundtruth_data = [], []
    label_y_predict = y_predict[1]
    p_predicted = y_predict[2]
    y_predict = y_predict[0]
    total_dis = 0
    mean_gro = np.mean(np.array(Y_data_test[0]))
    print(np.array(label_y_predict).shape)
    print(np.array(p_predicted).shape)
    print(np.array(label_Y_test).shape)
    print(np.array(y_predict).shape)
    print(np.array(Y_data_test[0]).shape)
    print(np.array(Y_data_test[1]).shape)
    TP, TN, FP, FN = 0, 0, 0, 0
    for i in range(0, len(y_predict)):
        label_y_test_l1 = np.array(Y_data_test[1][i])
        label_original.append(label_y_test_l1)
        label_predicted.append(label_y_predict[i])
        pj_predicted = np.array(p_predicted[i])
        one, two = [], []
        for j in range(0, len(pj_predicted)):
            for k in range(0, len(pj_predicted[j])):
                one.append(pj_predicted[j][k])
                two.append(label_Y_test[i][j][k])
        p_preicted_data.append(one)
        p_groundtruth_data.append(two)
        for j in range(0, len(y_predict[i])):
            if label_y_predict[i][j] >= 0.5:
                if label_y_test_l1[j] >= 0.5:
                    TP += 1
                else:
                    FN += 1
            else:
                if label_y_test_l1[j] >= 0.5:
                    FP += 1
                else:
                    TN += 1
            ab = abs(Y_data_test[0][i][j] - y_predict[i][j])
            cc = abs(y_predict[i][j] - mean_gro)
            total_dis += (cc * cc)
            if label_y_test_l1[j] == 1:
                total_ground_truth_high += Y_data_test[0][i][j]
                total_mae_high += ab
                total_mse_high += (ab * ab)
                aa = math.log(Y_data_test[0][i][j] + 1, 2) - math.log(y_predict[i][j] + 1, 2)
                total_msle_high += (aa * aa)
                if Y_data_test[0][i][j] == 0:
                    bb = 0
                else:
                    bb = abs((Y_data_test[0][i][j] - y_predict[i][j]) / Y_data_test[0][i][j])
                total_mape_high += bb
                length_high += 1
            else:
                total_ground_truth_normal += Y_data_test[0][i][j]
                total_mae_normal += ab
                total_mse_normal += (ab * ab)
                aa = math.log(Y_data_test[0][i][j] + 1, 2) - math.log(y_predict[i][j] + 1, 2)
                total_msle_normal += (aa * aa)
                if Y_data_test[0][i][j] == 0:
                    bb = 0
                else:
                    bb = abs((Y_data_test[0][i][j] - y_predict[i][j]) / Y_data_test[0][i][j])
                total_mape_normal += bb
                length_normal += 1

    MAE_high = total_mae_high / length_high
    MSE_high = total_mse_high / length_high
    MSLE_high= total_msle_high / length_high
    MAPE_high = total_mape_high / length_high
    ER_high = total_mae_high / total_ground_truth_high

    MAE_normal = total_mae_normal / length_normal
    MSE_normal = total_mse_normal / length_normal
    MSLE_normal = total_msle_normal / length_normal
    MAPE_normal = total_mape_normal / length_normal
    ER_normal = total_mae_normal / total_ground_truth_normal

    MAE_total = (total_mae_normal + total_mae_high) / (length_normal + length_high)
    MSE_total = (total_mse_normal + total_mse_high) / (length_normal + length_high)
    MSLE_total = (total_msle_normal + total_msle_high) / (length_normal + length_high)
    MAPE_total = (total_mape_normal + total_mape_high) / (length_normal + length_high)
    ER_total = (total_mae_normal + total_mae_high) / (total_ground_truth_normal + total_ground_truth_high)
    R2_total = 1 - ((total_mse_normal + total_mse_high) / total_dis)

    print("City: " + city)
    print("MAE TOTAL: " + str(MAE_total))
    print("MSE TOTAL: " + str(MSE_total))
    print("RMSE TOTAL: " + str(math.sqrt(MSE_total)))
    print("MSLE TOTAL: " + str(MSLE_total))
    print("MAPE TOTAL: " + str(MAPE_total))
    print("Error Rate TOTAL: " + str(ER_total))
    print("R-Squared TOTAL: " + str(R2_total))
    print(str(MAE_total) + " " + str(MSE_total) + " " + str(ER_total) + " " + str(MSLE_total) + " " +
          str(MAPE_total) + " " + str(math.sqrt(MSE_total)) + " " + str(R2_total))
    print("")
    print("MAE HIGH: " + str(MAE_high))
    print("MSE HIGH: " + str(MSE_high))
    print("RMSE HIGH: " + str(math.sqrt(MSE_high)))
    print("MSLE HIGH: " + str(MSLE_high))
    print("MAPE HIGH: " + str(MAPE_high))
    print("Error Rate HIGH: " + str(ER_high))
    print(str(MAE_high) + " " + str(MSE_high) + " " + str(ER_high) + " " + str(MSLE_high) + " " + str(MAPE_high) + " " + str(math.sqrt(MSE_high)))
    print("")
    print("MAE NORMAL: " + str(MAE_normal))
    print("MSE NORMAL: " + str(MSE_normal))
    print("RMSE NORMAL: " + str(math.sqrt(MSE_normal)))
    print("MSLE NORMAL: " + str(MSLE_normal))
    print("MAPE NORMAL: " + str(MAPE_normal))
    print("Error Rate NORMAL: " + str(ER_normal))
    print(str(MAE_normal) + " " + str(MSE_normal) + " " + str(ER_normal) + " " + str(MSLE_normal) + " " + str(
        MAPE_normal) + " " + str(math.sqrt(MSE_normal)))
    print(" ")


    if (TP + TN + FP + FN) == 0:
        acc = 0
    else:
        acc = (TP + TN)/(TP + TN + FP + FN)
    if (2 * TP + FP + FN) == 0:
        f1 = 0
    else:
        f1 = (2 * TP) / (2 * TP + FP + FN)
    if (TP + FP) == 0:
        precision = 0
    else:
        precision = TP / (TP + FP)
    if (TP + FN) == 0:
        recall = 0
    else:
        recall = TP / (TP + FN)
    if (TN + FP) == 0:
        specificity = 0
    else:
        specificity = TN / (TN + FP)
    print("ACC: " + str(acc))
    print("F1 score: " + str(f1))
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))
    print("Specificity: " + str(specificity))
    print(str(acc) + " " + str(f1) + " " + str(precision) + " " + str(recall) + " " + str(specificity))

    pyplot.plot(train_mse_all)
    pyplot.grid(b=True)
    pyplot.xlabel("Epochs", fontsize=18)
    pyplot.ylabel("Training MSE", fontsize=18)
    pyplot.legend()
    pyplot.show()

    pyplot.plot(val_mse_all)
    pyplot.grid(b=True)
    pyplot.xlabel("Epochs", fontsize=18)
    pyplot.ylabel("VAL MSE", fontsize=18)
    pyplot.legend()
    pyplot.show()

if __name__ == '__main__':
    tf.compat.v1.enable_eager_execution()
    # tf.config.experimental_run_functions_eagerly(True)
    with tf.device("gpu:0"):
        project_path = "D:\\ProgramProjects\\Python\\EVL\\"
        extreme_high_valid_percent, extreme_low_valid_percent = 0.80, 0.20
        main(project_path, "chicago bike 2021",
             "Chicago_BikePickUps_20210101_20211231_8Clusters.h5", "Chicago", extreme_high_valid_percent, extreme_low_valid_percent)
