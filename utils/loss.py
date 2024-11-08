import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss:
    """ Every loss subclass expects logits as input, regardless of model's output."""
    
    def __init__(self, rank, class_dist, output_type='logits'):
        if output_type=='logits':
            self.foo = lambda x: x
            self.bar = lambda x: F.log_softmax(x, dim=1)
        elif output_type=='logsoftmax':
            self.foo = lambda x: x
            self.bar = lambda x: torch.exp(x)
        elif output_type=='softmax':
            self.foo = lambda x: torch.log(x)
            self.bar = lambda x: x

        # CE loss guides model toward absolute frame-by-frame prediction correctness
        self.ce = nn.CrossEntropyLoss(weight=(1-class_dist/torch.sum(class_dist)).to(rank), reduction='mean')
        # MSE component punishes large variations intra-class predictions between consecutive time quants
        self.mse = nn.MSELoss(reduction='none')

    def __call__(self, i, predictions, ground_truth):        
        # CE + MSE loss metric tuning is taken from @BenjaminFiltjens's MS-GCN:
        # NOTE: subsegments have an overlap of 1 in outputs to enable correct MSE calculation, CE calculation should avoid double counting that frame
        ce = self.ce(self.foo(predictions if i==0 else predictions[:,:,1:]), ground_truth)

        predictions = self.bar(predictions)
        # in the reduced temporal resolution setting of the original model, MSE loss is expected to be large the higher
        # the receptive field since after that many frames a human could start performing a drastically diferent action
        mse = 0.15 * torch.mean(
            torch.clamp(
                self.mse(
                    predictions[:,:,1:],
                    predictions.detach()[:,:,:-1]),
                min=0,
                max=16))
    
        return ce, mse


class LossMultiStage(Loss):
    def __call__(self, i, predictions, ground_truth):
        ce_tot = 0
        mse_tot = 0

        for k in range(predictions.size(0)):
            ce, mse = super().__call__(i, predictions[k], ground_truth)
            ce_tot += ce
            mse_tot += mse

        return ce_tot, mse_tot
