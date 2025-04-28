import random,os,pickle,torch
from typing import Dict, Tuple

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import numpy as np
import pandas as pd
import pickle as pk

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
# tf.random.set_seed(142)
# np.random.seed(142)

# DataName="data_normalized"
DataName="data_2222_normalized"
modelname="WTK_Autoencoder_"+DataName


# tf.config.set_visible_devices([], 'GPU')
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

def reset_graph_and_set_seed(value: int = 42) -> None:
    # Reset the execution graph
    tf.keras.backend.clear_session()
    # Set seed
    # os.environ['PYTHONHASHSEED'] = str(value)
    # random.seed(value)
    # np.random.seed(value)
    # tf.random.set_seed(value)

def set_global_determinism(seed_value: int = 42) -> None:
    reset_graph_and_set_seed(value=seed_value)

    os.environ['TF_DETERMINISTIC_OPS'] = "1"
    os.environ['TF_CUDNN_DETERMINISTIC'] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

reset_graph_and_set_seed()


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


class Augmenetor:
    def __init__(self,modelname=modelname,MODELPATH="TFModels/"):
        with open(MODELPATH+"WTK_Autoencoder_model_argumens.pk","rb") as f:
           [self.n_rows,self.n_cols,self.nb_filters,self.nb_conv,self.code_size]=pk.load(f)

        self.model = WTK_Autoencoders(self.n_rows,self.n_cols,
                                 self.nb_filters,self.nb_conv,self.code_size)

        self.model.build((None, self.n_rows, self.n_cols))
        self.model.compile(loss=tf.keras.losses.Huber(), optimizer='adam', metrics=['mae', 'mse'])
        # self.model.build_graph().summary()
        self.model.load_weights(MODELPATH+modelname+".weights.h5")


        # tf.keras.utils.plot_model(model.build_graph(),
        #                           show_shapes=True,
        #                           expand_nested=True,
        #                           show_layer_activations=True)

    # create the negative pairs
    def negative_sample(self,x):
        enc_x0 = self.model.encoder.predict(x.reshape(1, 44, 3))
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
        xr = self.model.decoder.predict(enc_x1.reshape(1, 44, -1)).reshape(44, 3)

        xr = (xr - xr.min()) / (xr.max() - xr.min())

        return xr, 1 - x.reshape(1, 44, 3)


    def positive_sample(self,x):
        # Create the positive pairs
        enc_x0 = self.model.encoder.predict(x.reshape(1, 44, 3))
        enc_x1 = enc_x0
        std = np.std(enc_x0[enc_x0 > 0])
        mask = np.zeros_like(enc_x0)
        mask[np.ix_(*np.where(enc_x0 > 0))] = 1
        noise = np.random.randn() * std / 5 * mask
        xr = self.model.decoder.predict(enc_x1 + np.abs(noise))
        xr1 = xr.reshape(44, 3)
        xr = self.model.decoder.predict(enc_x0 * np.abs(enc_x0) > std)
        xr2 = xr.reshape(44, 3)
        xr1 = (xr1 - xr1.min()) / (xr1.max() - xr1.min())
        xr2 = (xr2 - xr2.min()) / (xr2.max() - xr2.min())
        return xr1, xr2

    def negative_sample_batch(self,x, verbose='auto'):
        enc_x0 = self.model.encoder.predict(x.reshape(-1, 44, 3),verbose=verbose)
        ind_selectlist,ind0list=[],[]
        for j in range(enc_x0.shape[0]):
            enc_x=enc_x0[j:j+1]
            std = np.std(enc_x[enc_x > 0])
            ind0 = np.where(enc_x > std / 2)
            ind1 = np.where(enc_x <= std / 2)
            ind_select = np.zeros_like(ind0)
            for i in range(len(ind1)):
                ind_tmp = ind1[i].copy()
                np.random.shuffle(ind_tmp)
                ind_tmp = ind_tmp[:len(ind0[0])]
                ind_select[i, :] = ind_tmp
            ind_select[0]=np.ones_like(ind_select[0])*j
            ind_selectlist.append(ind_select)
            ind0list.append((np.ones_like(ind0[0])*j,*ind0[1:]))
        enc_x1 = np.zeros_like(enc_x0)
        for ind_select,ind0 in zip(ind_selectlist,ind0list):
            for j in range(len(ind0[0])):
                enc_x1[ind_select[0][j], ind_select[1][j], ind_select[2][j]] = enc_x0[ind0[0][j], ind0[1][j], ind0[2][j]]
        xr = self.model.decoder.predict(enc_x1.reshape(-1, 44, enc_x0.shape[-1]),verbose=verbose)
        xr = (xr - xr.min()) / (xr.max() - xr.min())
        return xr, 1 - x

    def positive_sample_batch(self,x,verbose='auto'):
        enc_x0 = self.model.encoder.predict(x.reshape(-1, 44, 3),verbose=verbose)
        enc_x1 = enc_x0
        data1,data2=[],[]
        for i in range(enc_x0.shape[0]):
            enc_x=enc_x0[i:i+1]
            std = np.std(enc_x[enc_x > 0])
            mask = np.zeros_like(enc_x)
            mask[np.ix_(*np.where(enc_x > 0))] = 1
            noise = np.random.randn()* std / 5 * mask
            noise[np.isnan(noise)] = 0
            d1=enc_x1[i:i+1] + np.abs(noise)
            d2=enc_x * np.abs(enc_x) > std
            data1.append(d1[0])
            data2.append(d2[0])
        xr1 = self.model.decoder.predict(np.array(data1),verbose=verbose)
        # xr1 = xr.reshape(44, 3)
        xr2 = self.model.decoder.predict(np.array(data2),verbose=verbose)
        # xr2 = xr.reshape(44, 3)
        xr1 = (xr1 - xr1.min()) / (xr1.max() - xr1.min())
        xr2 = (xr2 - xr2.min()) / (xr2.max() - xr2.min())

        return xr1, xr2


    # Quantile dataset based on %5
    def data_quantile(self,df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[int, float], int]:
        r, b = pd.qcut(df.index,np.arange(0., 1.01, 0.05),retbins=True)
        sequence = {int(key): round(value, 2)
                    for key, value in zip(b.round(), np.arange(0., 1.05, 0.05))}
        windows_size = int(np.unique(np.diff(b.round()), return_counts=True)[0][-1])
        # df['Sequence'] = pd.Series(df.index.map(sequence), index=df.index).ffill()
        return df, sequence, windows_size


    def data_preProcessing(self,df: pd.DataFrame,
                           sequence: Dict[int, float],
                           windows_size: int,
                           #  is_train: bool=False
                           ) -> np.ndarray:
        # df = df.loc[:, features]
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

    # DataPath="../DATA/data/"
    # S1 = pd.read_csv(DataPath+"megas_small/cube_megas1_defect_1.csv",  names=['X', 'Y', 'Z'])
    # S2 = pd.read_csv(DataPath+"megas_small/cube_megas2_defect_1.csv",  names=['X', 'Y', 'Z'])

    def getProcessedData(self,S1):
      _, sequence, windows_size = self.data_quantile(S1)
      aug_data = self.data_preProcessing(S1, sequence, windows_size)
      return aug_data

    # plotting some of the negative pairs S1
    def PlotNagativeSample(self,aug_data):
        ind = np.random.randint(0,100,size=(1))[0]
        labels,data=[],[]
        labels.append(("X - True Value","Y - True Value","Z - True Value"))
        data.append(aug_data[ind])
        enc_x0 = self.model.encoder.predict(aug_data[ind].reshape(1,44,3))
        enc_x0 = enc_x0[0].reshape(1,44,self.nb_filters)
        xr = self.decoder.predict(enc_x0)
        xr = xr.reshape((44, 3))
        labels.append(("X - Fully Recovered by Decoder","Y - Fully Recovered by Decoder","Z - Fully Recovered by Decoder"))
        data.append(xr)
        # create the negative pairs
        x1, x2 = self.negative_sample(aug_data[ind])
        xr = x1.reshape((44, 3))
        labels.append(("X - Permuting Sparse Codes","Y - Permuting Sparse Codes","Z - Permuting Sparse Codes"))
        data.append(xr)
        xr = x2.reshape((44, 3))
        labels.append(("X - Negative Pair by Reversing","Y - Negative Pair by Reversing","Z - Negative Pair by Reversing"))
        data.append(xr)
        plt.figure(figsize=(16,12))
        for xr,labl in zip(data,labels):
          for dist, lab in zip(xr.T, labl):
            plt.plot(dist, label=lab)
        plt.legend()
        plt.xlabel("%5 Observations Sampled", fontsize=14)
        plt.ylabel("Amplitude", fontsize=14)
        plt.show()# create the negative pairs


    def PlotPositiveSamples(self,aug_data):
        # plotting some of the positive pairs S2
        plt.figure(figsize=(16,12))
        ind = np.random.randint(0,100,size=(1))[0]
        labels,data=[],[]
        labels.append(("X - True Value","Y - True Value","Z - True Value"))
        data.append(aug_data[ind])
        enc_x0 = self.model.encoder.predict(aug_data[ind].reshape(1,44,3))
        enc_x0 = enc_x0[0].reshape(1,44,self.nb_filters)
        xr = self.model.decoder.predict(enc_x0)
        data.append(xr.reshape((44, 3)))
        labels.append(("X - Fully Recovered by Decoder","Y - Fully Recovered by Decoder","Z - Fully Recovered by Decoder"))
        x1, x2 = self.positive_sample(aug_data[ind])
        data.append(x1.reshape((44, 3)))
        labels.append(("X - Augmented by adding Noise","Y - Augmented by adding Noise","Z - Augmented by adding Noise"))
        data.append(x2.reshape((44, 3)))
        labels.append(("X - Augmented by Thresholding","Y - Augmented by Thresholding","Z - Augmented by Thresholding"))

        for xr,labl in zip(data,labels):
          print(xr.shape)
          for dist, lab in zip(xr.T, labl):
            plt.plot(dist, label=lab)

        plt.legend()
        plt.xlabel("%5 Observations Sampled", fontsize=14)
        plt.ylabel("Amplitude", fontsize=14)
        plt.show()

    def getNegativeAugmentedDataBatch(self,S,concat=True):
        #  this will give the negative aumentation of the data
        if len(S.shape)==2:S=self.getProcessedData(S)
        x1, x2=self.negative_sample_batch(S,verbose=0)
        if concat:
            return np.concatenate([x1,x2])
        else:
            return x1,x2

    def getPositiveAugmentedDataBatch(self,S):
        #  this will give the postive aumentation of the data
        if len(S.shape)==2:S=self.getProcessedData(S)
        x1, x2=self.positive_sample_batch(S,verbose=0)
        return x1,x2

    def getSimclrAugmentation(self,device):
        #  this will give the postive aumentation of the data
        def getSimclrAugmentationFun(S,chunks=2):
            # It will do the augmentation to the Second half elements
            if chunks==2:
                S1, S2 = S.chunk(chunks)
                S2 = np.rollaxis(S2.cpu().numpy(), 2, 1)
                x1, x2 = self.positive_sample_batch(S2, verbose=0)
                S2aug = torch.Tensor(np.rollaxis(random.choice([x1,x2]), 2, 1)).to(device)
                return torch.cat((S1, S2aug))
            else:
                Scpu=S.cpu().numpy()
                Snp = np.rollaxis(Scpu, 2, 1)
                x1, x2 = self.positive_sample_batch(Snp,  verbose=0)
                S2aug=np.rollaxis(random.choice([x1,x2]), 2, 1)
                for i in range(len(Scpu)): # choose randomly data with orignal
                    if random.randint(0,10)<=6:S2aug[i]=Scpu[i]
                return torch.Tensor(S2aug).to(device)
        return getSimclrAugmentationFun,"WTK"

    def test(self,S):
        encoder=self.model.encoder.layers[0]
        out=encoder(self.getProcessedData(S)[:1])
        print(out.shape)


if __name__ == '__main__':
    aug=Augmenetor(modelname="WTK_Autoencoder_normal_laplace_data_normalized",MODELPATH="../TFModels/")
    print(aug.getPositiveAugmentedDataBatch(np.random.random((10,44,3)))[0].shape)