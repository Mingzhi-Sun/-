import numpy as np
import math
import gc
from keras.layers import Dense, Dropout, Flatten, Concatenate
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dropout, BatchNormalization
from keras.optimizers import adam_v2
from keras.callbacks import EarlyStopping
import netCDF4 as nc
from sklearn.preprocessing import StandardScaler
from keras import regularizers, Model, Input
from tqdm import tqdm
import tensorflow as tf
import os
import random



def r2_score(y_true, y_pred):
    numerator = ((y_true - y_pred) ** 2).sum()
    denominator = ((y_true - np.average(y_true)) ** 2).sum()
    r2=1-numerator/denominator
    return r2

def define_model(xx_train):
    # channel 1
    In_1 = Input(shape=(xx_train.shape[1], xx_train.shape[2], 1), dtype=tf.float16)

    model_1 = Conv2D(filters=8, kernel_size=3, strides=1, activation='tanh')(In_1)
    model_1 = MaxPooling2D(pool_size=2)(model_1)

    model_1 = Conv2D(filters=32, kernel_size=3,strides=1, activation='tanh')(model_1)
    model_1 = MaxPooling2D(pool_size=2)(model_1)

    model_1 = Flatten()(model_1)

    # channel 2
    In_2 = Input(shape=(xx_train.shape[1], xx_train.shape[2], 1), dtype=tf.float16)

    model_2 = Conv2D(filters=8, kernel_size=3, strides=1, activation='tanh')(In_2)
    model_2 = MaxPooling2D(pool_size=2)(model_2)

    model_2 = Conv2D(filters=32, kernel_size=3, strides=1, activation='tanh')(model_2)
    model_2 = MaxPooling2D(pool_size=2)(model_2)

    model_2 = Flatten()(model_2)

    # channel 3
    In_3 = Input(shape=(xx_train.shape[1],xx_train.shape[2], 1), dtype=tf.float16) #shape

    model_3 = Conv2D(filters=8, kernel_size=3, strides=1, activation='tanh')(In_3)
    model_3 = MaxPooling2D(pool_size=2)(model_3)

    model_3 = Conv2D(filters=32, kernel_size=3, strides=1, activation='tanh')(model_3)
    model_3 = MaxPooling2D(pool_size=2)(model_3)

    model_3 = Flatten()(model_3)

    # channel 4
    In_4 = Input(shape=(xx_train.shape[1], xx_train.shape[2], 1), dtype=tf.float16)#shape

    model_4 = Conv2D(filters=8, kernel_size=3, strides=1, activation='tanh')(In_4)
    model_4 = MaxPooling2D(pool_size=2)(model_4)

    model_4 = Conv2D(filters=32, kernel_size=3, strides=1, activation='tanh')(model_4)
    model_4 = MaxPooling2D(pool_size=2)(model_4)

    model_4 = Flatten()(model_4)

    # channel 5
    In_5 = Input(shape=(xx_train.shape[1], xx_train.shape[2], 1), dtype=tf.float16) #shape

    model_5 = Conv2D(filters=8, kernel_size=3, strides=1, activation='ReLU')(In_5)
    model_5 = MaxPooling2D(pool_size=2)(model_5)

    model_5 = Conv2D(filters=32, kernel_size=3, strides=1, activation='ReLU')(model_5)
    model_5 = MaxPooling2D(pool_size=2)(model_5)

    model_5 = Flatten()(model_5)

    # channel 6
    In_6 = Input(shape=(xx_train.shape[1], xx_train.shape[2], 1), dtype=tf.float16) #shape(model_6)

    model_6 = Conv2D(filters=8, kernel_size=3, strides=1, activation='ReLU')(In_6)
    model_6 = MaxPooling2D(pool_size=2)(model_6)

    model_6 = Conv2D(filters=32, kernel_size=3, strides=1, activation='ReLU')(model_6)
    model_6 = MaxPooling2D(pool_size=2)(model_6)

    model_6 = Flatten()(model_6)

    # channel 7
    In_7 = Input(shape=(xx_train.shape[1], xx_train.shape[2], 1), dtype=tf.float16)#shape(model_7)

    model_7 = Conv2D(filters=8, kernel_size=3, strides=1, activation='ReLU')(In_7)
    model_7 = MaxPooling2D(pool_size=2)(model_7)

    model_7 = Conv2D(filters=32, kernel_size=3, strides=1, activation='ReLU')(model_7)
    model_7 = MaxPooling2D(pool_size=2)(model_7)

    model_7 = Flatten()(model_7)

    # channel 8   Sediment 
    In_8 = Input(shape=(xx_train.shape[1], xx_train.shape[2], 1), dtype=tf.float16)  # shape(model_8)

    model_8 = Conv2D(filters=8, kernel_size=3, strides=1, activation='ReLU')(In_8)
    model_8 = MaxPooling2D(pool_size=2)(model_8)

    model_8 = Conv2D(filters=32, kernel_size=3, strides=1, activation='ReLU')(model_8)
    model_8 = MaxPooling2D(pool_size=2)(model_8)

    model_8 = Flatten()(model_8)

    # channel 9   Magnetic 
    In_9 = Input(shape=(xx_train.shape[1], xx_train.shape[2], 1), dtype=tf.float16)  # shape(model_9)

    model_9 = Conv2D(filters=8, kernel_size=3, strides=1, activation='ReLU')(In_9)
    model_9 = MaxPooling2D(pool_size=2)(model_9)

    model_9 = Conv2D(filters=32, kernel_size=3, strides=1, activation='ReLU')(model_9)
    model_9 = MaxPooling2D(pool_size=2)(model_9)

    model_9 = Flatten()(model_9)

    # channel 10   MDT 
    In_10 = Input(shape=(xx_train.shape[1], xx_train.shape[2], 1), dtype=tf.float16)  # shape(model_10)

    model_10 = Conv2D(filters=8, kernel_size=3, strides=1, activation='PReLU')(In_10)
    model_10 = MaxPooling2D(pool_size=2)(model_10)

    model_10 = Conv2D(filters=32, kernel_size=3, strides=1, activation='PReLU')(model_10)
    model_10 = MaxPooling2D(pool_size=2)(model_10)

    model_10 = Flatten()(model_10)

    # channel 11   ERR_MDT 
    In_11 = Input(shape=(xx_train.shape[1], xx_train.shape[2], 1), dtype=tf.float16)  # shape(model_11)

    model_11 = Conv2D(filters=8, kernel_size=3, strides=1, activation='PReLU')(In_11)
    model_11 = MaxPooling2D(pool_size=2)(model_11)

    model_11 = Conv2D(filters=32, kernel_size=3, strides=1, activation='PReLU')(model_11)
    model_11 = MaxPooling2D(pool_size=2)(model_11)

    model_11 = Flatten()(model_11)

    
    #combine
    merged = Concatenate()([model_1,
                            model_2,
                            model_3,
                            model_4,
                            model_5,
                            model_6,
                            model_7,
                            model_8,
                            model_9,
                            model_10,
                            model_11]) #merged
    dense1 = Dense(128,
                   activation='PReLU',
                   use_bias=True,
                   kernel_regularizer=regularizers.l1(0.0001))(merged) # interpretation
    dense2 = Dense(256,
                   activation='PReLU',
                   use_bias=True,
                   kernel_regularizer=regularizers.l1(0.0001))(dense1)  # interpretation
    dense3 = Dense(512,
                   activation='PReLU',
                   use_bias=True,
                   kernel_regularizer=regularizers.l1(0.0001))(dense2)  # interpretation
    output = Dense(1,
                   use_bias=True,
                   kernel_regularizer=regularizers.l1(0.0001))(dense3)

    Dropout(0.1)

    model = Model(inputs=[In_1, In_2, In_3, In_4, In_5, In_6, In_7, In_8, In_9, In_10, In_11],
                  outputs=output)
    # compile
    adam1 =adam_v2.Adam(learning_rate=0.0001)

    model.compile(loss='mse',
                  optimizer=adam1,
                  metrics=['mse'])
    return model

ifile = r'train_ship_GA.txt'  # ship-borne gravity anomaly
Data1 = nc.Dataset(r'top_0.25.nc')  # marine topography
Data2 = nc.Dataset(r'res_E_DOV.nc') # residual DOV for east
Data3 = nc.Dataset(r'res_N_DOV.nc') # residual DOV for north
Data4 = nc.Dataset(r'Sed_0.25.nc') # sediment
Data5 = nc.Dataset(r'magnetic_0.25.nc') # Magnetic
Data6 = nc.Dataset(r'MDT_0.25.nc') # MDT
Data7 = nc.Dataset(r'ERR_MDT_0.25.nc') # Err_MDT



h_gw = (Data1.variables["z"][:].data).T
e_gw = (Data2.variables["z"][:].data).T
n_gw = (Data3.variables["z"][:].data).T
s_gw = (Data4.variables["z"][:].data).T
m_gw = (Data5.variables["z"][:].data).T
MDT_gw = (Data6.variables["z"][:].data).T
ERR_MDT_gw = (Data7.variables["z"][:].data).T


Z1 = np.loadtxt(ifile) 

# Randomize the order of the data
random.shuffle(Z1)
random.shuffle(Z1)
random.shuffle(Z1)
random.shuffle(Z1)

# residual gravity anomaly for train
y_train = Z1[:, 2]

# inputs array for train
x_train = np.zeros((len(Z1[:, 1]), 64, 64, 11))

for i in tqdm(range(0, len(Z1[:, 1]))):
    # Note that the latitude and longitude Settings here should correspond to the input data, 
    #  not to the scope of the study area, which is generally larger than the scope of the study area.
    lon_num_min = math.floor((Z1[i, 0]-134)*60.0*4.0)-31
    lon_num_max = math.floor((Z1[i, 0]-134)*60.0*4.0)+33
    lat_num_min = math.floor((Z1[i, 1]-19)*60.0*4.0)-31
    lat_num_max = math.floor((Z1[i, 1]-19) * 60.0*4.0) + 33
    for k in range(0, 32):
        for l in range(0, 32):
            # Note that the latitude and longitude Settings here should correspond to the input data, 
            #  not to the scope of the study area, which is generally larger than the scope of the study area.
            x_train[i,  k, l, 0] = lon_num_min / 60.0 / 4.0 + 134 + 1.0 / 60.0 * 0.25 * k
            x_train[i,  k, l, 1] = lat_num_min / 60.0 / 4.0 + 19 + 1.0 / 60.0 * 0.25 * l
            x_train[i,  k, l, 2] = lon_num_min / 60.0 / 4.0 + 134 + 1.0 / 60.0 * 0.25 * k - Z1[i, 0]
            x_train[i,  k, l, 3] = lat_num_min / 60.0 / 4.0 + 19  + 1.0 / 60.0 * 0.25 * l - Z1[i, 1]
    x_train[i, 0:64, 0:64, 4] = n_gw[lon_num_min:lon_num_max, lat_num_min:lat_num_max]
    x_train[i, 0:64, 0:64, 5] = e_gw[lon_num_min:lon_num_max, lat_num_min:lat_num_max]
    x_train[i, 0:64, 0:64, 6] = h_gw[lon_num_min:lon_num_max, lat_num_min:lat_num_max]
    x_train[i, 0:64, 0:64, 7] = s_gw[lon_num_min:lon_num_max, lat_num_min:lat_num_max]
    x_train[i, 0:64, 0:64, 8] = m_gw[lon_num_min:lon_num_max, lat_num_min:lat_num_max]
    x_train[i, 0:64, 0:64, 9] = MDT_gw[lon_num_min:lon_num_max, lat_num_min:lat_num_max]
    x_train[i, 0:64, 0:64, 10] = ERR_MDT_gw[lon_num_min:lon_num_max, lat_num_min:lat_num_max]

del h_gw
del e_gw
del n_gw
del s_gw
del m_gw
gc.collect()

#inputs are standardized by removing the mean and scaling them to unit variance in each channel
mean_train = np.zeros((11))
std_train = np.zeros((11))
scaler = StandardScaler()
for i in range(0, 11):
    mean_train[i] = np.mean(x_train[:, :, :, i])
    std_train[i] = np.std(x_train[:, :, :, i])
    x_train[:, :, :, i] = (x_train[:, :, :, i]-mean_train[i])/std_train[i]
    print(mean_train[i], std_train[i])

print('Training...')
model = define_model(x_train)
model.fit([x_train[:, :, :, 0],
           x_train[:, :, :, 1],
           x_train[:, :, :, 2],
           x_train[:, :, :, 3],
           x_train[:, :, :, 4],
           x_train[:, :, :, 5],
           x_train[:, :, :, 6],
           x_train[:, :, :, 7],
           x_train[:, :, :, 8],
           x_train[:, :, :, 9],
           x_train[:, :, :, 10]],
          y_train,
          batch_size=254,
          shuffle=True,
          validation_split=0.05,
          epochs=20,
          callbacks=[EarlyStopping(monitor='mse', min_delta=0.05, patience=3)])

# save CNN model
model.save(r'GA_prediction_model.h5')

# calculate the r2_score for train
score_train = r2_score(y_train, np.transpose(model.predict([x_train[:, :, :, 0],
                                                            x_train[:, :, :, 1],
                                                            x_train[:, :, :, 2],
                                                            x_train[:, :, :, 3],
                                                            x_train[:, :, :, 4],
                                                            x_train[:, :, :, 5],
                                                            x_train[:, :, :, 6],
                                                            x_train[:, :, :, 7],
                                                            x_train[:, :, :, 8],
                                                            x_train[:, :, :, 9],
                                                            x_train[:, :, :, 10]])))
print(score_train)
print("successful!!!")




