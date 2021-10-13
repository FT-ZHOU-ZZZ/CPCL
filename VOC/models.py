import torchvision.models as models
from torch.nn import Parameter
from util import *
import torch
import torch.nn as nn


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class CouplingNetwork(nn.Module):
    def __init__(self):
        super(CouplingNetwork, self).__init__()
        self.fc = nn.Linear(in_features=2048, out_features=1024)

    def forward(self, features, semantics):
        """

        :param features:  # (n, c, d')
        :param semantics: # (n, c, d')
        :return: (n, c, d')
        """
        features_semantics = torch.mul(features, semantics)
        # features_semantics = features_semantics + features
        features_semantics = torch.cat((features_semantics, semantics),
                                       2).contiguous().view(-1, 2048)
        features_semantics = self.fc(features_semantics).contiguous().view(
            features.size(0), features.size(1), 1024)
        return features_semantics


class SemanticDecouple(nn.Module):
    """
    Semantic-Special Feature
    """
    def __init__(self,
                 num_classes,
                 feature_dim,
                 semantic_dim,
                 intermediary_dim=1024):
        super(SemanticDecouple, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.semantic_dim = semantic_dim
        self.intermediary_dim = intermediary_dim

        self.feature_trans = nn.Linear(self.feature_dim,
                                       self.intermediary_dim,
                                       bias=False)
        self.semantic_trans = nn.Linear(self.semantic_dim,
                                        self.intermediary_dim,
                                        bias=False)
        self.joint_trans = nn.Linear(self.intermediary_dim,
                                     self.intermediary_dim)

    def forward(self, global_feature, semantic_feature):
        """
        :param global_feature:  N*d
        :param semantic_feature:  C*k
        :return: N*C*d'
        """
        (n, d) = global_feature.shape
        (c, k) = semantic_feature.shape
        global_trans_feature = self.feature_trans(global_feature)
        semantic_trans_feature = self.semantic_trans(semantic_feature)
        global_trans_feature = global_trans_feature.unsqueeze(0).repeat(
            c, 1, 1).transpose(0, 1)
        semantic_trans_feature = semantic_trans_feature.unsqueeze(0).repeat(
            n, 1, 1)
        joint_trans_feature = torch.mul(
            global_trans_feature,
            semantic_trans_feature).contiguous().view(n * c, -1)
        semantic_special_feature = self.joint_trans(
            torch.tanh(joint_trans_feature)).contiguous().view(n, c, -1)
        return semantic_special_feature


class Model(nn.Module):
    def __init__(self,
                 model,
                 num_classes,
                 in_channel=300,
                 t=0.0,
                 adj_file=None):
        super(Model, self).__init__()
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )
        self.num_classes = num_classes
        self.pooling = nn.MaxPool2d(14, 14)

        # GCN module
        self.gc1 = GraphConvolution(in_channel, 512)
        self.gc2 = GraphConvolution(512, 1024)
        self.relu = nn.LeakyReLU(0.2)
        _adj = gen_A(num_classes, t, adj_file)
        self.A = Parameter(torch.from_numpy(_adj).float())

        # Semantic-special feature generation module
        self.semantic_special = SemanticDecouple(num_classes=self.num_classes,
                                                 feature_dim=2048,
                                                 semantic_dim=300,
                                                 intermediary_dim=1024)

        # Decoupling and coupling module
        self.decoupling_network = CouplingNetwork()
        self.coupling_network = CouplingNetwork()

        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, features, inp):
        features = self.features(features)
        features = self.pooling(features)
        features = features.contiguous().view(features.size(0), -1)

        inp = inp[0]  # (c, k)
        adj = gen_adj(self.A).detach()
        x = self.gc1(inp, adj)
        x = self.relu(x)
        semantic_features = self.gc2(x, adj)  # (c, d')
        semantic_special_features = self.semantic_special(features,
                                                          inp)  # (n, c, d')
        semantic_base_features = semantic_features.unsqueeze(0).repeat(
            semantic_special_features.size(0), 1, 1)  # (n, c, d')
        semantic_decoupling_features = self.coupling_network(
            semantic_special_features, semantic_base_features)
        semantic_coupling_features = self.decoupling_network(
            semantic_special_features, semantic_base_features)

        decoupling_distance = torch.norm(semantic_special_features -
                                         semantic_decoupling_features,
                                         dim=2)
        coupling_distance = torch.norm(semantic_special_features -
                                       semantic_coupling_features,
                                       dim=2)
        return decoupling_distance, coupling_distance

    def get_config_optim(self, lr, lrp):
        return [
            {
                'params': self.features.parameters(),
                'lr': lr * lrp
            },
            {
                'params': self.gc1.parameters(),
                'lr': lr
            },
            {
                'params': self.gc2.parameters(),
                'lr': lr
            },
            {
                'params': self.decoupling_network.parameters(),
                'lr': lr
            },
            {
                'params': self.coupling_network.parameters(),
                'lr': lr
            },
            {
                'params': self.semantic_special.parameters(),
                'lr': lr
            },
        ]


def get_model(num_classes, t, pretrained=True, adj_file=None, in_channel=300):
    model = models.resnet101(pretrained=pretrained)
    return Model(model,
                 num_classes,
                 t=t,
                 adj_file=adj_file,
                 in_channel=in_channel)
