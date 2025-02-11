import jittor as jt
from jittor import nn
import torch.nn.functional as F




class Upsample(nn.Module):
    def __init__(self, scale_factor, mode='nearest'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def execute(self, x):
        return nn.interpolate(x, scale_factor=self.scale_factor,
                             mode = self.mode)


class AdaptiveInstanceNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm1d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = None
        self.bias = None
        self.register_buffer('running_mean', jt.zeros(num_features))
        self.register_buffer('running_var', jt.ones(num_features))

    def execute(self, x):
        assert self.weight is not None and \
               self.bias is not None, "Please assign AdaIN weight first"
        b, c = x.size(0), x.size(1)  # batch size & channels
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])
        out = nn.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            False, self.momentum, self.eps) #zxy_edit改为False测试
        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'



def get_acti_layer(acti='relu'):

    if acti == 'relu':
        return [lambda x: nn.relu(x)]
    elif acti == 'lrelu':
        return [lambda x: nn.leaky_relu(x,scale=0.2)]
    elif acti == 'tanh':
        return [lambda x: nn.Tanh(x)]
    elif acti == 'none':
        return []
    else:
        assert 0, "Unsupported activation: {}".format(acti)


def get_norm_layer(norm='none', norm_dim=None):
    

    if norm == 'bn':
        return [nn.BatchNorm1d(norm_dim)]
    elif norm == 'in':
        # return [nn.InstanceNorm1d(norm_dim, affine=False)]  # for rt42!
        return [nn.InstanceNorm1d(norm_dim, affine=True)]
    elif norm == 'adain':
        # print("norm_dim:", norm_dim) # norm_dim: 144
        return [AdaptiveInstanceNorm1d(norm_dim)]
    elif norm == 'none':
        return []
    else:
        assert 0, "Unsupported normalization: {}".format(norm)


def get_dropout_layer(dropout=None):
    if dropout is not None:
        return [nn.Dropout(p=dropout)]
    else:
        return []




def LinearBlock(in_dim, out_dim, dropout=None, norm='none', acti='relu'):

    use_bias = True
    layers = []
    layers.append(nn.Linear(in_dim, out_dim, bias=use_bias))
    layers += get_dropout_layer(dropout)
    layers += get_norm_layer(norm, norm_dim=out_dim)
    layers += get_acti_layer(acti)

    return layers


