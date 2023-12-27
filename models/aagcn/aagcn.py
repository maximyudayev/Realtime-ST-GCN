import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import Graph, LayerNorm, BatchNorm1d
from models.stgcn.stgcn import StgcnLayer


class Model(nn.Module):
    def __init__(self, **kwargs):

        super(Model, self).__init__()

        conf = kwargs['aa-gcn']

        # load graph
        # A is the normalized non-learnable adjacency matrix
        self.graph = Graph(strategy=kwargs['strategy'], **kwargs['graph'])
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = kwargs['graph']['num_node']
        temporal_kernel_size = conf['kernel']
        kernel_size = (temporal_kernel_size, spatial_kernel_size)

        self.num_classes = kwargs['num_classes']

        self.streams = nn.ModuleList([nn.ModuleDict({
            "norm_in": LayerNorm([kwargs['in_feat'], 1, A.size(1)], device=0) if kwargs['normalization'] == 'LayerNorm' else BatchNorm1d(kwargs['in_feat'] * A.size(1), track_running_stats=False, device=0),
            "fcn_in": nn.Conv2d(
                in_channels=conf['in_feat'],
                out_channels=conf['in_ch'][0],
                kernel_size=1,
                device=0 if stream == 'joints' else 2),
            "gcn_networks": nn.ModuleList([
                AgcnLayer(
                    in_channels=conf['in_ch'][i],
                    out_channels=conf['out_ch'][i],
                    kernel_size=kernel_size,
                    partitions=A.size(0),
                    stride=conf['stride'][i],
                    residual=not not conf['residual'][i],
                    dropout=conf['dropout'][i],
                    num_joints=kwargs['graph']['num_node'],
                    importance=conf['importance'],
                    normalization=kwargs['normalization'],
                    device=0 if i < 5 else 1 if stream == 'joints' else 2 if i < 5 else 3)
                for i in range(conf['layers'])]),
            "fcn_out": nn.Conv2d(
                in_channels=conf['out_ch'][-1],
                out_channels=kwargs['num_classes'],
                kernel_size=1,
                device=1 if stream == 'joints' else 3)
        }) for stream in ['joints','bones']])

        if kwargs['output_type']=='logits':
            self.probability = lambda x: x
        elif kwargs['output_type']=='logsoftmax':
            self.probability = lambda x: F.log_softmax(x, dim=1)
        elif kwargs['output_type']=='softmax':
            self.probability = lambda x: F.softmax(x, dim=1)


    def forward(self, x_joint):
        self.A.to(0)
        # turns joint coordinates into bone coordinate vectors, pointing from source to target joint
        # gets the "far" immediately-connected joints per each joint (row)
        A_far = self.graph.get_adjacency_raw()[2].astype(bool)
        # each bone identified by the target joint index, same size adjacency matrix as for joint data, but COG bone is empty vector
        x_bone = torch.zeros_like(x_joint)
        # TODO: slice into tensor using tensor of indices, for efficient vectorized setting w/o looping
        for i in range(self.graph.num_node):
            x_bone[:,:,:,A_far[i]] = x_joint[:,:,:,A_far[i]] - x_joint[:,:,:,i,None]
        x_bone.to(2)

        # pass both streams through the coresponding branch and add prediction probabilities
        x_joint = self.streams[0]['norm_in'](x_joint)
        # remap the features to the network size
        x_joint = self.streams[0]['fcn_in'](x_joint)
        # forward pass through GCN layers
        for i, gcn in enumerate(self.streams[0]['gcn_networks']):
            if i == 5:
                x_joint.to(1)
                self.A.to(1)
            x_joint = gcn(x_joint, self.A)
        # global pooling (across time L, and nodes V)
        x_joint = F.avg_pool2d(x_joint, x_joint.size()[2:])
        # prediction
        x_joint = self.streams[0]['fcn_out'](x_joint)
        x_joint = self.probability(x_joint.squeeze(-1))

        # pass both streams through the coresponding branch and add prediction probabilities
        x_bone = self.streams[1]['norm_in'](x_bone)
        # remap the features to the network size
        x_bone = self.streams[1]['fcn_in'](x_bone)
        # forward pass through GCN layers
        for i, gcn in enumerate(self.streams[1]['gcn_networks']):
            if i == 5:
                x_bone.to(3)
                self.A.to(3)
            x_bone = gcn(x_bone, self.A)
        # global pooling (across time L, and nodes V)
        x_bone = F.avg_pool2d(x_bone, x_bone.size()[2:])
        # prediction
        x_bone = self.streams[1]['fcn_out'](x_bone)
        x_bone = self.probability(x_bone.squeeze(-1))

        # NOTE: original 2s-AGCN sums probabilities of both streams, not logits
        return x_bone + x_joint.to(3)


class AgcnLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        partitions,
        stride,
        residual,
        dropout,
        num_joints,
        importance,
        normalization='LayerNorm',
        device=None):

        super(AgcnLayer, self).__init__()

        coeff_embedding = 4

        self.embedding_channels = out_channels//coeff_embedding
        self.partitions = partitions
        self.num_joints = num_joints

        # fully-learnable adjacency matrix B, initialized as 0's
        self.B = nn.Parameter(torch.zeros(partitions, num_joints, num_joints, device=device), requires_grad=True) if importance else 1

        # self-attention adjacency matrix C, produces 2 matrices
        self.theta = nn.Conv2d(in_channels, self.embedding_channels*partitions, 1, device=device)
        self.phi = nn.Conv2d(in_channels, self.embedding_channels*partitions, 1, device=device)

        # beyond more complicated adjacency matrix, exactly same as regular ST-GCN (Yan, et. al, 2018).
        self.st_gcn = StgcnLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            partitions=partitions,
            num_joints=num_joints,
            stride=stride,
            dropout=dropout,
            residual=residual,
            normalization=normalization,
            device=device)


    def forward(self, x, A):
        N,_,L,_ = x.size()
        # (N,C,L,V -> N,P,C'*L,V)
        theta = self.theta(x).view(N, self.partitions, self.embedding_channels*L, self.num_joints).permute(0,1,3,2)
        phi = self.phi(x).view(N, self.partitions, self.embedding_channels*L, self.num_joints)
        # (N,P,V,V)
        C = F.softmax(torch.matmul(theta, phi), dim=3)

        # graph convolution
        x = self.st_gcn(x, A+self.B+C)

        return x
