from abc import *
import torch.nn as nn

def getActivatation(name):
    if name=="relu":
        return nn.ReLU()
    elif name=="relu6":
        return nn.ReLU6()
    elif name=="tanh":
        return nn.Tanh()

class BaseModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, last_dim, num_classes=10, activation="relu", simclr_dim=128):
        super(BaseModel, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(last_dim, num_classes),
        )
        self.simclr_layer = nn.Sequential(
            nn.Linear(last_dim, last_dim),
            getActivatation(activation),
            nn.Linear(last_dim, simclr_dim),
        )
        self.shift_cls_layer = nn.Linear(last_dim, 2)
        self.joint_distribution_layer = nn.Linear(last_dim, 4 * num_classes)
        self.shift_dis = nn.Sequential(
            nn.Linear(simclr_dim * 2, simclr_dim),
            nn.ReLU(),
            nn.Linear(simclr_dim, 1),
            nn.Sigmoid()
        )
        self.softmax = nn.Softmax()
        self.Sigmoid = nn.Sigmoid()

    @abstractmethod
    def penultimate(self, inputs, all_features=False):
        pass

    def forward(self, inputs=None, penultimate=False, simclr=False, shift=False, joint=False,softmax=False,sigmoid=False):
        try:
            _aux = {}
            _return_aux = False

            features = self.penultimate(inputs)
            output = self.linear(features)
            if softmax: output=self.softmax(output)
            elif sigmoid:output=self.Sigmoid(output)
            if penultimate:
                _return_aux = True
                _aux['penultimate'] = features

            if simclr:
                _return_aux = True
                _aux['simclr'] = self.simclr_layer(features)

            if shift:
                _return_aux = True
                _aux['shift'] = self.shift_cls_layer(features)

            if joint:
                _return_aux = True
                _aux['joint'] = self.joint_distribution_layer(features)


            if _return_aux:
                return output, _aux

            return output
        except Exception as e:
            if "x" in str(e):
                stre=str(e).split("x")[1].split()
                print("Issue seems in last_dim change last dim from",stre[-1],"to",stre[0])
            raise e
