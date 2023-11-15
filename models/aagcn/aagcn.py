import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import Graph
from models.stgcn.stgcn import StgcnLayer


class Model(nn.Module):
    def __init__(self, rank, **kwargs):

        super(Model, self).__init__()

        conf = kwargs['aa-gcn']

        # load graph
        # A is the normalized non-learnable adjacency matrix
        self.graph = Graph(strategy=kwargs['strategy'], **kwargs['graph'])
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = kwargs['graph']['num_node']
        temporal_kernel_size = conf['kernel'][0]
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        
        self.segment = kwargs['segment']
        self.num_classes = kwargs['num_classes']
        self.rank = rank

        # input capture normalization
        # (N,C,L,V)
        self.is_bn = kwargs['is_bn']

        if self.is_bn:
            self.data_bn = nn.BatchNorm1d(kwargs['in_feat'] * A.size(1), track_running_stats=kwargs['is_bn_stats'])

        self.streams = nn.ModuleList([nn.ModuleDict({
            "fcn_in": nn.Conv2d(
                in_channels=conf['in_feat'], 
                out_channels=conf['in_ch'][0][0], 
                kernel_size=1),
            "gcn_networks": nn.ModuleList([
                module for sublist in [[AgcnLayer(
                    rank=rank,
                    in_channels=conf['in_ch'][i][j],
                    out_channels=conf['out_ch'][i][j],
                    kernel_size=kernel_size,
                    partitions=A.size(0),
                    stride=conf['stride'][i][j],
                    residual=not not conf['residual'][i][j],
                    dropout=conf['dropout'][i][j],
                    num_joints=kwargs['graph']['num_node'],
                    importance=conf['importance'],
                    segment=kwargs['segment'],
                    receptive_field=kwargs['receptive_field'],
                    is_bn=kwargs['is_bn'],
                    track_running_stats=kwargs.get('is_bn_stats'))
                for j in range(layers_in_stage)] 
                for i, layers_in_stage in enumerate(conf['layers'])] for module in sublist]),
            "fcn_out": nn.Conv2d(
                in_channels=conf['out_ch'][-1][-1],
                out_channels=kwargs['num_classes'],
                kernel_size=1)
        }) for _ in ['joints','bones']])

        if kwargs['stream_sum']=='logits':
            self.probability = lambda x: x
        elif kwargs['probability']=='logsoftmax':
            self.probability = lambda x: F.log_softmax(x, dim=1)
        else:
            self.probability = lambda x: F.softmax(x, dim=1)


    def forward(self, x_joint):
        # placeholder for model's output between two streams
        output = torch.zeros(self.segment, self.num_classes, 1, device=self.rank)

        # turns joint coordinates into bone coordinate vectors, pointing from source to target joint
        # gets the "far" immediately-connected joints per each joint (row)
        A_far = self.graph.get_adjacency_raw()[2].astype(bool)
        # each bone identified by the target joint index, same size adjacency matrix as for joint data, but COG bone is empty vector
        x_bone = torch.zeros_like(x_joint)
        # TODO: slice into tensor using tensor of indices, for efficient vectorized setting w/o looping
        for i in range(self.graph.num_node):
            x_bone[:,:,:,A_far[i]] = x_joint[:,:,:,A_far[i]] - x_joint[:,:,:,i,None]

        # pass both streams through the coresponding branch and add prediction probabilities
        for data, modules in zip((x_joint, x_bone), self.streams):
            if self.is_bn:
                # data normalization
                N, C, T, V = data.size()
                # permutes must copy the tensor over as contiguous because .view() needs a contiguous tensor
                # this incures extra overhead
                data = data.permute(0, 3, 1, 2).contiguous()
                # (N,V,C,T)
                data = data.view(N, V * C, T)
                data = self.data_bn(data)
                data = data.view(N, V, C, T)
                data = data.permute(0, 2, 3, 1)
                # (N,C,T,V)

            # remap the features to the network size
            data = modules['fcn_in'](data)
            
            # forward pass through GCN layers
            for gcn in modules['gcn_networks']:
                data = gcn(data, self.A)
            
            # global pooling (across time L, and nodes V
            data = F.avg_pool2d(data, data.size()[2:])

            # prediction
            data = modules['fcn_out'](data)

            # NOTE: original 2s-AGCN sums probabilities of both streams, not logits
            output += self.probability(data.squeeze(-1))

        return output


class AgcnLayer(nn.Module):
    def __init__(
        self, 
        rank,
        in_channels, 
        out_channels, 
        kernel_size, 
        partitions, 
        stride, 
        residual, 
        dropout,
        num_joints,
        importance,
        segment,
        receptive_field,
        is_bn=True,
        track_running_stats=True):

        super(AgcnLayer, self).__init__()

        coeff_embedding = 4

        self.embedding_channels = out_channels//coeff_embedding
        self.segment = segment
        self.receptive_field = receptive_field
        self.partitions = partitions
        self.num_joints = num_joints

        # fully-learnable adjacency matrix B, initialized as 0's
        self.B = nn.Parameter(torch.zeros(partitions, num_joints, num_joints, device=rank), requires_grad=True) if importance else 1

        # self-attention adjacency matrix C, produces 2 matrices
        self.theta = nn.Conv2d(in_channels, self.embedding_channels*partitions, 1)
        self.phi = nn.Conv2d(in_channels, self.embedding_channels*partitions, 1)

        # beyond more complicated adjacency matrix, exactly same as regular ST-GCN (Yan, et. al, 2018).
        self.st_gcn = StgcnLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            partitions=partitions,
            stride=stride,
            dropout=dropout,
            residual=residual,
            is_bn=is_bn,
            track_running_stats=track_running_stats)


    def forward(self, x, A):
        L = x.size(2)
        # (N,C,L,V -> N,P,C'*L,V)
        theta = self.theta(x).view(self.segment, self.partitions, self.embedding_channels*L, self.num_joints).permute(0,1,3,2)
        phi = self.phi(x).view(self.segment, self.partitions, self.embedding_channels*L, self.num_joints)
        # (N,P,V,V)
        C = F.softmax(torch.matmul(theta,phi), dim=3)
        
        # graph convolution
        x = self.st_gcn(x, A+self.B+C)

        return x
