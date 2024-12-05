import torch
import torch.nn as nn

from lib.lorentz.manifold import CustomLorentz

class LorentzAct(nn.Module):
    """ Implementation of a general Lorentz Activation on space components. 
    """
    def __init__(self, activation, manifold: CustomLorentz):
        super(LorentzAct, self).__init__()
        self.manifold = manifold
        self.activation = activation # e.g. torch.relu

    def forward(self, x):
        return self.manifold.lorentz_activation(x, self.activation)
    

class LorentzReLU(nn.Module):
    """ Implementation of Lorentz ReLU Activation on space components. 
    """
    def __init__(self, manifold: CustomLorentz):
        super(LorentzReLU, self).__init__()
        self.manifold = manifold

    def forward(self, x):
        return self.manifold.lorentz_relu(x)


class LorentzGlobalAvgPool2d(torch.nn.Module):
    """ Implementation of a Lorentz Global Average Pooling based on Lorentz centroid defintion. 
    """
    def __init__(self, manifold: CustomLorentz, keep_dim=False):
        super(LorentzGlobalAvgPool2d, self).__init__()

        self.manifold = manifold
        self.keep_dim = keep_dim

    def forward(self, x):
        """ x has to be in channel-last representation -> Shape = bs x H x W x C """
        bs, h, w, c = x.shape
        x = x.view(bs, -1, c)
        x = self.manifold.centroid(x)
        if self.keep_dim:
            x = x.view(bs, 1, 1, c)

        return x
