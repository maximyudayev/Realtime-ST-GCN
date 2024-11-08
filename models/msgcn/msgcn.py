import torch
import torch.nn as nn
import torch.nn.functional as F
from models.stgcn import Model as Stgcn
from models.mstcn.mstcn import SingleStage as MsTcnStage
from torch.nn import DataParallel as DP


class Model(nn.Module):
    """
    """

    def __init__(self, **kwargs):
        super().__init__()

        self.stages = kwargs['stages']
        self.num_classes = kwargs['num_classes']

        conf = kwargs['ms-tcn']

        self.generator_stage = Stgcn(**kwargs)

        self.refinement_stages = nn.ModuleList([
            MsTcnStage(
                in_channels=kwargs['num_classes'],
                out_channels=kwargs['num_classes'],
                num_filters=conf['filters'][i],
                num_layers=conf['layers'][i],
                kernel=conf['kernel'][i],
                dropout=conf['dropout'][i]) for i in range(conf['stages'])])
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

        if torch.cuda.device_count() > 1:
            self.generator_stage = DP(self.generator_stage)


    def forward(self, x):
        N,_,_,_ = x.size()
        device = x.get_device()

        outputs = torch.zeros(self.stages, 1, self.num_classes, N, device=device)

        x = self.generator_stage(x)
        # (N,C,1) -> (1,C,L,1) | N=L
        x = x.view(1, N, self.num_classes, 1).permute(0,2,1,3)
        outputs[0] = self.out(x[:,:,:,0])

        for i, stage in enumerate(self.refinement_stages):
            x = stage(self.probability(x))
            outputs[i+1] = self.out(x.squeeze(-1))
        return outputs
