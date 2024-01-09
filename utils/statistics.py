import torch


class Statistics:
    def __call__(self, i, predictions, ground_truth):
        # this only sums the number of top-1 correctly predicted frames, but doesn't look at prediction jitter
        # NOTE: avoids double counting single overlapping frame occuring in two consequent subsegments
        predictions = predictions if i==0 else predictions[:,:,1:]

        _, top5_predicted = torch.topk(predictions, k=5, dim=1)
        top1_predicted = top5_predicted[:,0,:]
        # top5_probs[0,:,torch.bitwise_and(torch.any(top5_predicted == labels[:,None,:], dim=1), top1_predicted != labels)[0]].permute(1,0) # probabilities of classes where top-1 and top-5 don't intersect
        top1_cor = torch.sum(top1_predicted == ground_truth).data.item()
        top5_cor = torch.sum(top5_predicted == ground_truth[:,None]).data.item()
        tot = ground_truth.numel()

        return top1_predicted, top5_predicted, top1_cor, top5_cor, tot


class StatisticsMultiStage(Statistics):
    def __call__(self, i, predictions, ground_truth):
        return super().__call__(i, predictions[-1], ground_truth)
