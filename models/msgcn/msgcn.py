import torch
import torch.nn as nn
import torch.nn.functional as F
from models.stgcn import Model as Stgcn
from models.mstcn.mstcn import SingleStage as MsTcnStage

class Model(nn.Module):
    """
    """

    def __init__(self, rank, **kwargs):
        super().__init__()

        self.stages = kwargs['stages']
        self.num_classes = kwargs['num_classes']
        self.segment = kwargs['segment']
        self.rank = rank

        conf = kwargs['ms-tcn']

        self.generator_stage = Stgcn(rank, **kwargs)

        self.refinement_stages = nn.ModuleList([
            MsTcnStage(
                in_channels=kwargs['num_classes'],
                out_channels=kwargs['num_classes'],
                num_filters=conf['filters'][i],
                num_layers=conf['layers'][i],
                kernel=conf['kernel'][i],
                dropout=conf['dropout'][i]) for i in range(1, conf['stages'])])
        
        if kwargs['refine']=='logits':
            self.probability = lambda x: x
        elif kwargs['probability']=='logsoftmax':
            self.probability = lambda x: F.log_softmax(x, dim=1)
        else:
            self.probability = lambda x: F.softmax(x, dim=1)

    def forward(self, x):
        outputs = torch.zeros(self.stages, 1, self.num_classes, self.segment, device=self.rank)

        x = self.generator_stage(x)
        # (N,C,1) -> (1,C,L,1) | N=L
        x = x.permute(2,1,0,3)
        outputs[0] = x[:,:,:,0]

        for i, stage in enumerate(self.refinement_stages):
            x = stage(self.probability(x))
            outputs[i+1] = x.squeeze(-1)
        return outputs
