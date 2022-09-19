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

from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
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
from imblearn.over_sampling import SMOTE


def rSquared(true, predicted):
    cols = predicted.shape[1]
    rsq = np.zeros(shape=(cols), dtype=np.float32)
    for j in range(cols):
        rsq[j] = r2_score(true[:, j], predicted[:, j])
    return rsq


###################
# 加载数据
###################
preprocessed_DNAMeth = pd.read_csv('C:\\Users\\MR Wang\\Desktop\\test\\tt1\\LIHC_preprocessed_DNAMeth.csv')
preprocessed_RNASeq = pd.read_csv('C:\\Users\\MR Wang\\Desktop\\test\\tt1\\LIHC_preprocessed_RNASeq.csv')
preprocessed_CNA = pd.read_csv('C:\\Users\\MR Wang\\Desktop\\test\\tt1\\LIHC_preprocessed_CNA.csv')
labels = pd.read_csv('C:\\Users\\MR Wang\\Desktop\\test\\tt1\\LIHC_labels.csv')

x1 = preprocessed_DNAMeth
x2 = preprocessed_CNA
y = preprocessed_RNASeq



# 连接五种特征
x1 = pd.DataFrame(x1)
x2 = pd.DataFrame(x2)
df = [x1,x2]
z = pd.concat(df,axis=1)

# 将数据拆分为训练和测试数据集
x_train, x_test, y_train, y_test, labels_train, labels_test = train_test_split(z, y,  labels, test_size=0.2)

# 在 [0-1] 范围内缩放数据
scalar = MinMaxScaler()
x_train = scalar.fit_transform(x_train)
x_test = scalar.transform(x_test)

# 添加高斯噪音
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(0.0, 1.0, x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(0.0, 1.0, x_test.shape)

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# 设置输入和输出神经元的编号
num_in_neurons = z.shape[1]

with tf.device('/gpu:0'):
    # this is the size of our encoded representations
    encoding_dim1 = 500
    encoding_dim2 = 200

    lambda_act = 0.0001
    lambda_weight = 0.001
    # this is our input placeholder
    input_data = Input(shape=(num_in_neurons,))
    # first encoded representation of the input
    encoded = Dense(encoding_dim1, activation='relu', activity_regularizer=regularizers.l1(lambda_act),
                    kernel_regularizer=regularizers.l2(lambda_weight), name='encoder1')(input_data)
    # second encoded representation of the input
    encoded = Dense(encoding_dim2, activation='relu', activity_regularizer=regularizers.l1(lambda_act),
                    kernel_regularizer=regularizers.l2(lambda_weight), name='encoder2')(encoded)
    # first lossy reconstruction of the input
    decoded = Dense(encoding_dim1, activation='relu', name='decoder1')(encoded)
    # the final lossy reconstruction of the input
    decoded = Dense(num_in_neurons, activation='sigmoid', name='decoder2')(decoded)

    # this model maps an input to its reconstruction
    autoencoder = Model(inputs=input_data, outputs=decoded)

    myencoder = Model(inputs=input_data, outputs=encoded)
    autoencoder.compile(optimizer='sgd', loss='mse')
    # training
    print('training the autoencoder')
    print('training the autoencoder')
    history = autoencoder.fit(x_train_noisy, x_train,
                              epochs=25,
                              batch_size=8,
                              shuffle=True,
                              validation_data=(x_test_noisy, x_test)
                              )
    autoencoder.trainable = False  # freeze autoencoder weights

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

ae_train = myencoder.predict(x_train)
ae_test = myencoder.predict(x_test)


print(type(ae_train))
print(type(ae_test))
print("done")