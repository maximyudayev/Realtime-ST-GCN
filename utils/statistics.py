import torch
import torch.nn.functional as F


class Statistics:
    def __init__(self, probability='logsoftmax', is_foo=True):
        if not is_foo:
            self.foo = lambda x: x
        elif probability=='logsoftmax':
            self.foo = lambda x: F.log_softmax(x, dim=1)
        else:
            self.foo = lambda x: F.softmax(x, dim=1)

    def __call__(self, i, predictions, ground_truth):
        # `foo` transforms logit predictions into probabilities
        predictions = self.foo(predictions[:,:,1 if i!=0 else 0:])
        # this only sums the number of top-1 correctly predicted frames, but doesn't look at prediction jitter
        # NOTE: avoid double counting single overlapping frame occuring in two consequent segments
        # NOTE: best to use the same probability function (softmax or log_softmax) as loss to get same absolute values to print model's prediction confidence per class, but not important for metric counting
        top5_probs, top5_predicted = torch.topk(predictions, k=5, dim=1)
        top1_predicted = top5_predicted[:,0,:]
        # top5_probs[0,:,torch.bitwise_and(torch.any(top5_predicted == labels[:,None,:], dim=1), top1_predicted != labels)[0]].permute(1,0) # probabilities of classes where top-1 and top-5 don't intersect
        top1_cor = torch.sum(top1_predicted == ground_truth).data.item()
        top5_cor = torch.sum(top5_predicted == ground_truth[:,None]).data.item()
        tot = ground_truth.numel()
    
        return top1_predicted, top5_predicted, top1_cor, top5_cor, tot


class StatisticsMultiStage(Statistics):
    def __call__(self, i, predictions, ground_truth):
        return super().__call__(i, predictions[-1], ground_truth)


class StatisticsOneToOneMultiStage(StatisticsMultiStage):
    def __call__(self, i, predictions, ground_truth):
        return super().__call__(0, predictions, ground_truth)
