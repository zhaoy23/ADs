import os,torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import BaseModel
import pickle as pk

def getActivatation(name):
    if name=="relu":
        return nn.ReLU()
    elif name=="relu6":
        return nn.ReLU6()
    elif name=="tanh":
        return nn.Tanh()

class CustomModel(BaseModel):
    # This is the custom model used both in uncertianity and contrastive.
    def __init__(self,num_classes,nconv=1,activatation1="relu",activatation2="relu") -> None:
        with open("TFModels/WTK_Autoencoder_model_argumens.pk", "rb") as f:
            [n_rows, n_cols, nb_filters, nb_conv, code_size] = pk.load(f)
        # this is calulate bease on other value like filter, padding strides
        self.last_dim=159
        super(CustomModel, self).__init__(self.last_dim, num_classes)
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.nb_conv=nb_conv
        # self.nb_conv=self.last_dim
        # define convolution layers
        self.conv1=nn.Conv1d(n_cols, nb_filters, kernel_size=self.nb_conv,  padding=self.nb_conv//2-1, bias=False)
        self.activation1=getActivatation(activatation1)
        self.convlist=[nn.Conv1d(nb_filters, nb_filters, kernel_size=self.nb_conv, padding=self.nb_conv//2, bias=False) for i in range(nconv)]
        self.activation2=getActivatation(activatation2)


        for i,x in enumerate(self.convlist):
            self.__setattr__("conv"+str(i+2),x)

    def penultimate(self, x, all_features=False):
        out_list = []
        out = self.activation1(self.conv1(x))
        for conv in self.convlist:
            out = self.activation2(conv(out))

        out = F.avg_pool1d(out, self.nb_conv)
        out = out.view(out.size(0), -1)
        if all_features:
            return out, out_list
        else:
            return out
        return out



