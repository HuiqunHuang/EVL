from __future__ import print_function

import tensorflow as tf
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Activation, Dense, GRU
from tensorflow.python.keras.models import Model
from tensorflow.python.layers.base import Layer
from tensorflow.keras import initializers

def MultiTimeSeriesPredictionModel(categorynum=4, len_closeness=3, len_day=7, T=8):
    main_inputs = []
    outputs = []

    category_input = Input(shape=(len_closeness, categorynum))
    main_inputs.append(category_input)
    label_input = Input(shape=(len_day, categorynum))
    main_inputs.append(label_input)
    extreme_data_input = Input(shape=(len_day, T, categorynum))
    main_inputs.append(extreme_data_input)

    '''
        gru+dense modeling component
    '''
    gru_units, dense_units = 128, 64
    h_c_s = GRU(gru_units, input_shape=(len_closeness, categorynum), activation="relu")(category_input)
    print("h_c_s: " + str(h_c_s))
    h_c_s = Dense(dense_units, input_shape=(len_closeness, gru_units), activation="relu")(h_c_s)
    print("h_c_s: " + str(h_c_s))
    h_c_s = Dense(categorynum, input_shape=(dense_units, ), activation="relu")(h_c_s)
    print("h_c_s: " + str(h_c_s))


    '''
        extreme data modeling component, extreme high and normal only
    '''
    evm = ExtremeValueModelling(len_day=len_day, category_num=categorynum)
    extreme_data_input = Activation(activation="tanh")(extreme_data_input)
    main_output, ut, p = evm.extreme_value_modeling(extreme_data_input, label_input, h_c_s, h_c_s)

    main_output = Activation('relu', name='main_output')(main_output)


    outputs.append(main_output)
    outputs.append(ut)
    outputs.append(p)
    print("main_output: " + str(main_output))
    print("ut: " + str(ut))
    print("p: " + str(p))

    model = Model(main_inputs, outputs)

    return model

class ExtremeValueModelling(Layer):
    def __init__(self, len_day, category_num):
        """
        p : feature dimension
        h0 : initial hidden state
        c0 : initial cell state
        """
        super(ExtremeValueModelling, self).__init__(name="ExtremeValueModelling")
        self.len_day = len_day
        self.category_num = category_num
        self.initial_state = None
        # self.gru = GRU(category_num, return_state=True, activation="relu")
        self.gru = GRU(category_num, return_state=True, activation="sigmoid")
        self.b = self.add_weight(name="b", shape=tf.TensorShape([self.category_num]),
                                 initializer=initializers.RandomNormal(mean=1, stddev=1), trainable=True)

    def extreme_value_modeling(self, data_input, label_input, ht, last_output):
        data_input = tf.unstack(data_input, self.len_day, 1)
        label_input = tf.unstack(label_input, self.len_day, 1)


        all_h, all_output, all_p = [], [], []
        ct = []
        ct_sum = None
        for i in range(0, self.len_day):
            output, h_c_s = self.gru(data_input[i], initial_state=self.initial_state)
            self.reset_state(h_c_s)
            ctj = tf.math.multiply(ht, h_c_s)

            ct.append(ctj)
            if ct_sum is None:
                ct_sum = ctj
            else:
                ct_sum = tf.add(ct_sum, ctj)
            all_h.append(h_c_s)
            all_output.append(output)
            pj = h_c_s
            all_p.append(pj)
        all_p = tf.stack(all_p, 1)
        print("all_p: " + str(all_p))
        ut = None
        for i in range(0, self.len_day):
            atj = tf.math.divide_no_nan(ct[i], ct_sum)
            utj = tf.math.multiply(atj, label_input[i])
            if ut is None:
                ut = utj
            else:
                ut = tf.add(ut, utj)

        output = last_output + ut * self.b
        return output, ut, all_p

    def reset_state(self, h_c_s):
        self.initial_state = h_c_s

