import os,torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import BaseModel
import pickle as pk

class CustomModel(BaseModel):
    # This is the custom model used both in uncertianity and contrastive.
    def __init__(self,num_classes) -> None:
        with open("TFModles/WTK_Autoencoder_model_argumens.pk", "rb") as f:
            [n_rows, n_cols, nb_filters, nb_conv, code_size] = pk.load(f)
        # this is calulate bease on other value like filter, padding strides
        self.last_dim=159
        super(CustomModel, self).__init__(self.last_dim, num_classes)
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.nb_conv=nb_conv
        # define convolution layers
        self.conv1=nn.Conv1d(n_cols, nb_filters, kernel_size=nb_conv,  padding=nb_conv//2-1, bias=False)
        self.conv2= nn.Conv1d(nb_filters, nb_filters, kernel_size=nb_conv, padding=nb_conv//2, bias=False)

    def penultimate(self, x, all_features=False):
        out_list = []
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))

        out = F.avg_pool1d(out, self.nb_conv)
        out = out.view(out.size(0), -1)
        if all_features:
            return out, out_list
        else:
            return out
        return out


