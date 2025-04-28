# Commented out IPython magic to ensure Python compatibility.
import random
import pickle
from collections import OrderedDict
from typing import List, Text, Dict, Tuple, Optional, Union
import os

import numpy as np
import pandas as pd
import pickle as pk

from sklearn.preprocessing import StandardScaler

import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Input,
                                     Dense,
                                     Dropout,
                                     Activation,
                                     Flatten,
                                     Reshape,
                                     Layer,
                                     Lambda)

from tensorflow.keras.layers import (Convolution1D,
                                     MaxPooling1D,
                                     UpSampling1D,
                                     Conv1D)

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import non_neg

import matplotlib.pyplot as plt

dtype="_normalized"
# dtype=""
modelpath="../TFModels/"
DataPath="../../DATA/data"+dtype+"/"
modelname="WTK_Autoencoder"+dtype
plt.style.use('fivethirtyeight')
plt.rcParams["figure.figsize"] = [16, 12]


"""## Reproductioned setup"""


def reset_graph_and_set_seed(value: int = 42) -> None:
    """
    the function is responsible for generate specific random
    generated sequence for the reproducibility. and also responsible for
    resetting the graph execution of the tensorflow.
    Args:
      value[int]: seed value. Default 42
    """
    # Reset the execution graph
    tf.keras.backend.clear_session()

    # Set seed
    os.environ['PYTHONHASHSEED'] = str(value)
    random.seed(value)
    np.random.seed(value)
    tf.random.set_seed(value)

def set_global_determinism(seed_value: int = 42) -> None:
    reset_graph_and_set_seed(value=seed_value)

    os.environ['TF_DETERMINISTIC_OPS'] = "1"
    os.environ['TF_CUDNN_DETERMINISTIC'] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)


reset_graph_and_set_seed()

"""## Data Ingestion"""


# Quantile dataset based on %5
def data_quantile(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[int, float], int]:
    # this function add the sequence of 0 to 1 with incex of .05
    # retrun calculate window size  and also sequence
    r, b = pd.qcut(df.index,
                   np.arange(0., 1.01, 0.05),
                   retbins=True)
    sequence = {int(key): round(value, 2)
                for key, value in zip(b.round(), np.arange(0., 1.05, 0.05))}
    windows_size = int(np.unique(np.diff(b.round()), return_counts=True)[0][-1])
    df['Sequence'] = pd.Series(df.index.map(sequence), index=df.index).ffill()
    return df, sequence, windows_size


def data_augmentaion(df: pd.DataFrame,
                     sequence: Dict[int, float],
                     windows_size: int,
                     #  is_train: bool=False
                     features
                     ) -> np.ndarray:
    df = df.loc[:, features]
    # Select data from %5 to %90
    # quantile_5th = list(sequence.keys())[1] # start of the %5
    qualtile_90th = list(sequence.keys())[-3]  # start of the %90
    df = df.iloc[:qualtile_90th, :]

    timestamp = []

    for i in range(len(df)):
        # find the end of this pattern
        end_ix = i + windows_size
        # making the forecast days underlying the window size
        # out_end_ix = end_ix + forecast_windows_size
        if end_ix > len(df) - 1:
            break
        seq_x = df.iloc[i:end_ix, :]

        # Adding the sequences to the lists
        timestamp.append(seq_x)

    return np.stack(timestamp, axis=0)


def readAndAugumen(filepath,windows_size=44):
    # it return data in windows
    S = pd.read_csv(filepath, names=['X', 'Y', 'Z'])
    df, sequence, _ = data_quantile(S)
    features = [col for col in S.columns if col not in ("Sequence",)]
    data = S.loc[:, features]
    aug_data = data_augmentaion(data, sequence, windows_size,features)
    return aug_data



trainDataAug = np.concatenate([readAndAugumen(DataPath+"megas_small/"+f) for f in os.listdir(DataPath+"megas_small")])

trainDataAug = np.concatenate([trainDataAug]+[readAndAugumen(DataPath+"megax_large/"+f) for f in os.listdir(DataPath+"megax_large")])

"""## Data Preparation"""


def data_preparation(aug_data: np.ndarray, batch_size: int = 1) -> tf.data.Dataset:
    """
    Preparing the numpy array for training
    """
    input_data = (aug_data, aug_data)
    dataset = tf.data.Dataset.from_tensor_slices(input_data)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


"""$$14 => D \in R^{53 \times 14}$$

### Model Architecture
"""


class WTK_Autoencoders(tf.keras.Model):
    def __init__(self,
                 n_rows: int,
                 n_cols: int,
                 nb_filters: int,
                 nb_conv: int,
                 code_size: int) -> None:
        super().__init__()
        reset_graph_and_set_seed()
        self.n_rows = n_rows
        self.n_cols = n_cols
        # (1) Encoder
        self.encoder = Sequential([
            Conv1D(nb_filters,
                   (nb_conv),
                   activation='relu',
                   padding='same',
                   input_shape=(n_rows, n_cols)),
            Conv1D(nb_filters,
                   (nb_conv),
                   activation='relu',
                   padding='same'),
        ], name='Encoder')

        self.pool_shape = self.encoder.output_shape

        self.encoder.add(Lambda(function=self.wtall,
                                output_shape=self.pool_shape[1:],
                                name='WTALL_Layer'))

        # (2) Decoder
        self.decoder = Sequential([
            Conv1D(n_cols,
                   (code_size),
                   strides=1,
                   padding='same',
                   input_shape=self.pool_shape[1:],
                   kernel_constraint=non_neg()),
            # Flatten()
        ], name='Decoder')

    def call(self, inputs):
        encoder = self.encoder(inputs)
        decoder = self.decoder(encoder)
        return decoder

    # winner takes all layer: only keep the highest value
    def wtall(self, X):
        M = K.max(X, axis=(1), keepdims=True)
        R = K.switch(K.equal(X, M), X, K.zeros_like(X))
        return R

    # similar to the sequential or functional API like.
    def build_graph(self):
        x = Input(shape=(self.n_rows, self.n_cols))
        return Model(inputs=[x], outputs=self.call(x), name='WTK_Autoencoders')

"""### hyperparameters"""

nb_epoch = 200
n_rows,n_cols,nb_filters,nb_conv,code_size=44,3,53,14,14
print("Parameters:-",n_rows,n_cols,nb_filters,nb_conv,code_size)
model = WTK_Autoencoders(n_rows, n_cols,nb_filters, nb_conv, code_size)

model.build((None, n_rows, n_cols))
model.compile(loss=tf.keras.losses.Huber(), optimizer='adam', metrics=['mae', 'mse'])

model.build_graph().summary()
# tf.keras.utils.plot_model(model.build_graph(),
#                           show_shapes=True,
#                           expand_nested=True,
#                           show_layer_activations=True,
#                           to_file="WTK_Architecure.png")


with open(modelpath+"WTK_Autoencoder_model_argumens.pk", "wb") as f:
    pk.dump([n_rows, n_cols, nb_filters, nb_conv, code_size], f)

"""### Train & Evaluate"""

train_data = data_preparation(trainDataAug, 16)

# training the model
ES = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                      patience=5,
                                      mode='min',
                                      verbose=1)

if os.path.exists(modelpath+modelname+".weights.h5"):
    print("Found PreTrained Model")
    model.load_weights(modelpath+modelname+".weights.h5")
else:
    os.makedirs(modelpath,exist_ok=True)
    history = model.fit(train_data,
                        epochs=500,
                        # callbacks=[ES],
                        shuffle=False,
                        verbose=1)
    model.save_weights(modelpath+modelname+".weights.h5")
    # tf.keras.models.save_model(model, modelpath+modelname)

    # Training curve
    plt.plot(history.history['loss'])
    plt.show()

# The learned dictionaries
# Model Training Complete You can remove the exit to check Augmented Data
exit()
W = np.asarray(K.eval(model.decoder.layers[0].weights[0]))

print(W.shape)
fig = plt.figure(figsize=(16, 12))
for i in range(min(50, nb_filters)):
    plt.subplot(10, 5, i + 1)
    plt.plot(W[:, i, 0])

fig.text(0.5, 0, 'Decoder Kernel Depth', ha='center')
fig.text(0, 0.5, 'Decoder Kernel Weights', va='center', rotation='vertical')

"""### Save the model

## Testing Negative-positive pairs outputs

### Negative Sampling
"""


# create the negative pairs
def negative_sample(x, enc, dec):
    enc_x0 = enc.predict(x.reshape(1, 44, 3))
    enc_x1 = np.zeros_like(enc_x0)

    std = np.std(enc_x0[enc_x0 > 0])

    ind0 = np.where(enc_x0 > std / 2)
    ind1 = np.where(enc_x0 <= std / 2)
    ind_select = np.zeros_like(ind0)

    for i in range(len(ind1)):
        ind_tmp = ind1[i].copy()
        np.random.shuffle(ind_tmp)
        ind_tmp = ind_tmp[:len(ind0[0])]
        ind_select[i, :] = ind_tmp
    for j in range(len(ind0[0])):
        enc_x1[ind_select[0][j], ind_select[1][j], ind_select[2][j]] = enc_x0[ind0[0][j], ind0[1][j], ind0[2][j]]
    xr = dec.predict(enc_x1.reshape(1, 44, -1)).reshape(44, 3)

    xr = (xr - xr.min()) / (xr.max() - xr.min())

    return xr, 1 - x.reshape(1, 44, 3)


aug_data = readAndAugumen(DataPath+"megas_small/cube_megas1_defect_1.csv")
s2_aug_data =  readAndAugumen(DataPath+"megas_small/cube_megas2_defect_1.csv")

s2_aug_data.shape

"""#### S1"""

# plotting some of the negative pairs S1
plt.figure(figsize=(16, 12))
ind = np.random.randint(0, 100, size=(1))[0]

labels = ("X - True Value",
          "Y - True Value",
          "Z - True Value")

for dist, label in zip(aug_data[ind].T, labels):
    plt.plot(dist, label=label)

enc_x0 = model.encoder.predict(aug_data[ind].reshape(1, 44, 3))
enc_x0 = enc_x0[0].reshape(1, 44, nb_filters)
xr = model.decoder.predict(enc_x0)
xr = xr.reshape((44, 3))

labels = ("X - Fully Recovered by Decoder",
          "Y - Fully Recovered by Decoder",
          "Z - Fully Recovered by Decoder")

for dist, label in zip(xr.T, labels):
    plt.plot(dist, label=label)

# create the negative pairs
x1, x2 = negative_sample(aug_data[ind], model.encoder, model.decoder)

xr = x1.reshape((44, 3))
labels = ("X - Permuting Sparse Codes",
          "Y - Permuting Sparse Codes",
          "Z - Permuting Sparse Codes")
for dist, label in zip(xr.T, labels):
    plt.plot(dist, label=label)

xr = x2.reshape((44, 3))
labels = ("X - Negative Pair by Reversing",
          "Y - Negative Pair by Reversing",
          "Z - Negative Pair by Reversing")

for dist, label in zip(xr.T, labels):
    plt.plot(dist, label=label)

plt.legend()
plt.xlabel("%5 Observations Sampled", fontsize=14)
plt.ylabel("Amplitude", fontsize=14)
plt.show()  # create the negative pairs

"""#### S2"""

# plotting some of the negative pairs S2
plt.figure(figsize=(16, 12))
ind = np.random.randint(0, 100, size=(1))[0]

labels = ("X - True Value",
          "Y - True Value",
          "Z - True Value")

for dist, label in zip(s2_aug_data[ind].T, labels):
    plt.plot(dist, label=label)

enc_x0 = model.encoder.predict(s2_aug_data[ind].reshape(1, 44, 3))
enc_x0 = enc_x0[0].reshape(1, 44, nb_filters)
xr = model.decoder.predict(enc_x0)
xr = xr.reshape((44, 3))

labels = ("X - Fully Recovered by Decoder",
          "Y - Fully Recovered by Decoder",
          "Z - Fully Recovered by Decoder")

for dist, label in zip(xr.T, labels):
    plt.plot(dist, label=label)

x1, x2 = negative_sample(s2_aug_data[ind], model.encoder, model.decoder)

xr = x1.reshape((44, 3))
labels = ("X - Permuting Sparse Codes",
          "Y - Permuting Sparse Codes",
          "Z - Permuting Sparse Codes")
for dist, label in zip(xr.T, labels):
    plt.plot(dist, label=label)

xr = x2.reshape((44, 3))
labels = ("X - Negative Pair by Reversing",
          "Y - Negative Pair by Reversing",
          "Z - Negative Pair by Reversing")

for dist, label in zip(xr.T, labels):
    plt.plot(dist, label=label)

plt.legend()
plt.xlabel("%5 Observations Sampled", fontsize=14)
plt.ylabel("Amplitude", fontsize=14)
plt.show()

"""### Positive Pairs"""


# Create the positive pairs

def positive_sample(x, enc, dec):
    enc_x0 = enc.predict(x.reshape(1, 44, 3))

    enc_x1 = enc_x0
    std = np.std(enc_x0[enc_x0 > 0])

    mask = np.zeros_like(enc_x0)
    mask[np.ix_(*np.where(enc_x0 > 0))] = 1

    noise = np.random.randn() * std / 5 * mask
    xr = dec.predict(enc_x1 + np.abs(noise))
    xr1 = xr.reshape(44, 3)

    xr = dec.predict(enc_x0 * np.abs(enc_x0) > std)
    xr2 = xr.reshape(44, 3)

    xr1 = (xr1 - xr1.min()) / (xr1.max() - xr1.min())
    xr2 = (xr2 - xr2.min()) / (xr2.max() - xr2.min())

    return xr1, xr2


"""#### S1"""

# plotting some of the positive pairs S2
plt.figure(figsize=(16, 12))
ind = np.random.randint(0, 100, size=(1))[0]

labels = ("X - True Value",
          "Y - True Value",
          "Z - True Value")

for dist, label in zip(aug_data[ind].T, labels):
    plt.plot(dist, label=label)

enc_x0 = model.encoder.predict(aug_data[ind].reshape(1, 44, 3))
enc_x0 = enc_x0[0].reshape(1, 44, nb_filters)
xr = model.decoder.predict(enc_x0)
xr = xr.reshape((44, 3))

labels = ("X - Fully Recovered by Decoder",
          "Y - Fully Recovered by Decoder",
          "Z - Fully Recovered by Decoder")

for dist, label in zip(xr.T, labels):
    plt.plot(dist, label=label)

x1, x2 = positive_sample(aug_data[ind], model.encoder, model.decoder)

xr = x1.reshape((44, 3))
labels = ("X - Augmented by adding Noise",
          "Y - Augmented by adding Noise",
          "Z - Augmented by adding Noise")
for dist, label in zip(xr.T, labels):
    plt.plot(dist, label=label)

xr = x2.reshape((44, 3))
labels = ("X - Augmented by Thresholding",
          "Y - Augmented by Thresholding",
          "Z - Augmented by Thresholding")

for dist, label in zip(xr.T, labels):
    plt.plot(dist, label=label)

plt.legend()
plt.xlabel("%5 Observations Sampled", fontsize=14)
plt.ylabel("Amplitude", fontsize=14)
plt.show()

"""#### S2"""

# plotting some of the positive pairs S2
plt.figure(figsize=(16, 12))
ind = np.random.randint(0, 100, size=(1))[0]

labels = ("X - True Value",
          "Y - True Value",
          "Z - True Value")

for dist, label in zip(s2_aug_data[ind].T, labels):
    plt.plot(dist, label=label)

enc_x0 = model.encoder.predict(s2_aug_data[ind].reshape(1, 44, 3))
enc_x0 = enc_x0[0].reshape(1, 44, nb_filters)
xr = model.decoder.predict(enc_x0)
xr = xr.reshape((44, 3))

labels = ("X - Fully Recovered by Decoder",
          "Y - Fully Recovered by Decoder",
          "Z - Fully Recovered by Decoder")

for dist, label in zip(xr.T, labels):
    plt.plot(dist, label=label)

x1, x2 = positive_sample(s2_aug_data[ind], model.encoder, model.decoder)

xr = x1.reshape((44, 3))
labels = ("X - Augmented by adding Noise",
          "Y - Augmented by adding Noise",
          "Z - Augmented by adding Noise")
for dist, label in zip(xr.T, labels):
    plt.plot(dist, label=label)

xr = x2.reshape((44, 3))
labels = ("X - Augmented by Thresholding",
          "Y - Augmented by Thresholding",
          "Z - Augmented by Thresholding")

for dist, label in zip(xr.T, labels):
    plt.plot(dist, label=label)

plt.legend()
plt.xlabel("%5 Observations Sampled", fontsize=14)
plt.ylabel("Amplitude", fontsize=14)
plt.show()