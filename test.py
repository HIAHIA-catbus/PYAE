import os

os.environ['PYTHONHASHSEED'] = '0'
# 以下是在定义良好的初始状态下启动 Numpy 生成的随机数所必需的。
import numpy as np

np.random.seed(42)
# 以下是启动核心 Python 生成的随机数处于明确定义的状态所必需的.
import random as rn

rn.seed(12345)

# tf.set_random_seed() 将使 TensorFlow 后端中的随机数生成具有明确定义的初始状态。
import tensorflow as tf

tf.random.set_seed(1234)

# 强制tensorflow使用单线程
# session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
#                               inter_op_parallelism_threads=1)
# from keras import backend as K
# sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
# K.set_session(sess)

from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras import regularizers
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import r2_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def rSquared(true, predicted):
    cols = predicted.shape[1]
    rsq = np.zeros(shape=(cols), dtype=np.float32)
    for j in range(cols):
        rsq[j] = r2_score(true[:, j], predicted[:, j])
    return rsq


###################
# 加载数据
###################
preprocessed_DNAMeth = pd.read_csv('G:\\TCGA\\data\\data_pre_with_symble\\LIHC_preprocessed_DNAMeth.csv')
preprocessed_RNASeq = pd.read_csv('G:\\TCGA\\data\\data_pre_with_symble\\LIHC_preprocessed_RNASeq.csv')
preprocessed_CNA = pd.read_csv('G:\\TCGA\\data\\data_pre_with_symble\\LIHC_preprocessed_CNA.csv')
preprocessed_PSI = pd.read_csv('G:\\TCGA\\data\\data_pre_with_symble\\LIHC_preprocessed_PSI.csv')
preprocessed_MIRNA = pd.read_csv('G:\\TCGA\\data\\data_pre_with_symble\\LIHC_preprocessed_MIRNA.csv')
labels = pd.read_csv('G:\\TCGA\\data\\data_pre_with_symble\\LIHC_labels.csv')

x1 = preprocessed_DNAMeth
x2 = preprocessed_RNASeq
x3 = preprocessed_CNA
x4 = preprocessed_PSI
x5 = preprocessed_MIRNA

# 连接五种特征
x1 = pd.DataFrame(x1)
x2 = pd.DataFrame(x2)
x3 = pd.DataFrame(x3)
x4 = pd.DataFrame(x4)
x5 = pd.DataFrame(x5)
df = [x1,x2,x3,x4,x5]
z = pd.concat(df,axis=1)

z.to_csv('C:\\Users\\MR Wang\\Desktop\\ae_full.csv',index=False)


# 将数据拆分为训练和测试数据集
x_train, x_test, labels_train, labels_test = train_test_split(z,  labels, test_size=0.2)

# # 在 [0-1] 范围内缩放数据
# scalar = MinMaxScaler()
# x_train = scalar.fit_transform(x_train)
# x_test = scalar.transform(x_test)

# #添加高斯噪音
# noise_factor = 0.5
# x_train_noisy = x_train + noise_factor * np.random.normal(0.0, 1.0, x_train.shape)
# x_test_noisy = x_test + noise_factor * np.random.normal(0.0, 1.0, x_test.shape)
#
# x_train_noisy = np.clip(x_train_noisy, 0., 1.)
# x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# 设置输入和输出神经元的编号
num_in_neurons = z.shape[1]

# AE提取特征
with tf.device('/gpu:0'):
    # 多层AE各层数量
    encoding_dim1 = 500
    encoding_dim2 = 200

    lambda_act = 0.0001
    lambda_weight = 0.001
    # 输入大小
    input_data = Input(shape=(num_in_neurons,))

    # 第一层编码器
    encoded = Dense(encoding_dim1, activation='relu', activity_regularizer=regularizers.l1(lambda_act), kernel_regularizer=regularizers.l2(lambda_weight), name='encoder1')(input_data)
    # 第二层编码器
    encoded = Dense(encoding_dim2, activation='relu', activity_regularizer=regularizers.l1(lambda_act), kernel_regularizer=regularizers.l2(lambda_weight), name='encoder2')(encoded)

    # 第一层解码器
    decoded = Dense(encoding_dim1, activation='relu', name='decoder1')(encoded)
    # 第二层解码器
    decoded = Dense(num_in_neurons, activation='sigmoid', name='decoder2')(decoded)


    # 映射
    autoencoder = Model(inputs=input_data, outputs=decoded)

    myencoder = Model(inputs=input_data, outputs=encoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    # autoencoder.compile(optimizer='sgd', loss='mse')
    # training
    print('training the autoencoder')
    history = autoencoder.fit(x_train,x_train,
                              epochs=20,
                              batch_size=16,
                              shuffle=True,
                              verbose=1,
                              validation_data=(x_test,x_test)
                              )
    # autoencoder.trainable = False  # freeze autoencoder weights
    ae_train = myencoder.predict(x_train)
    ae_test = myencoder.predict(x_test)


    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss  20 16')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()
print("dsf")