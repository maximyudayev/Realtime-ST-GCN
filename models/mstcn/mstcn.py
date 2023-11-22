import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(
        self,
        rank,
        **kwargs):

        super().__init__()

        conf = kwargs['ms-tcn']
        self.stages = conf['stages']
        self.num_classes = kwargs['num_classes']
        self.segment = kwargs['segment']
        self.rank = rank

        self.generator_stage = SingleStage(
                in_channels=conf['in_feat'],
                out_channels=kwargs['num_classes'],
                num_filters=conf['filters'][0],
                num_layers=conf['layers'][0],
                kernel=conf['kernel'][0],
                dropout=conf['dropout'][0])
        
        self.refinement_stages = nn.ModuleList([SingleStage(
                in_channels=kwargs['num_classes'],
                out_channels=kwargs['num_classes'],
                num_filters=conf['filters'][i],
                num_layers=conf['layers'][i],
                kernel=conf['kernel'][i],
                dropout=conf['dropout'][i]) for i in range(1, conf['stages'])])

        if kwargs['refine']=='logits':
            self.probability = lambda x: x
        elif kwargs['refine']=='logsoftmax':
            self.probability = lambda x: F.log_softmax(x, dim=1)
        elif kwargs['refine']=='softmax':
            self.probability = lambda x: F.softmax(x, dim=1)

        if kwargs['output_type']=='logits':
            self.out = lambda x: x
        elif kwargs['output_type']=='logsoftmax':
            self.out = lambda x: F.log_softmax(x, dim=1)
        elif kwargs['output_type']=='softmax':
            self.out = lambda x: F.softmax(x, dim=1)


    def forward(self, x):
        # NOTE: original implementation passes probabilities to refinement stages, not logits
        outputs = torch.zeros(self.stages, 1, self.num_classes, self.segment, device=self.rank)
        
        x = self.generator_stage(x)
        # (1,C,L,V)
        # pool features at the output of the generators stage across the joint dimension
        x = F.avg_pool2d(x, (1, x.size(-1)))
        # (1,C,L,1)
        outputs[0] = self.out(x.squeeze(-1))

        for i, stage in enumerate(self.refinement_stages):
            x = stage(self.probability(x))
            outputs[i+1] = self.out(x.squeeze(-1))
        return outputs


class SingleStage(nn.Module):
    def __init__(
        self, 
        in_channels=None, 
        out_channels=None, 
        num_filters=64, 
        num_layers=10, 
        kernel=3, 
        dropout=0.0):

        super().__init__()

        self.conv_in = nn.Conv2d(in_channels, num_filters, 1)
        self.layers = nn.Sequential(
            *[DilatedResidualLayer(
                in_channels=num_filters, 
                out_channels=num_filters,
                kernel=kernel,
                dilation=2**i,
                dropout=dropout) for i in range(num_layers)])
        self.conv_out = nn.Conv2d(num_filters, out_channels, 1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.layers(x)
        x = self.conv_out(x)
        return x


class DilatedResidualLayer(nn.Module):
    def __init__(
        self, 
        in_channels=64, 
        out_channels=64, 
        kernel=3, 
        dilation=1, 
        dropout=0.5):

        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (kernel,1), padding=(dilation,0), dilation=(dilation,1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1),
            nn.Dropout(dropout, inplace=True))

    def forward(self, x):
        out = self.conv(x)
        return x + out
