import torch
from .metric import Metric
import pandas as pd


class F1Score(Metric):
    """Computes segmental F1@k score with an IoU threshold, proposed by Lea, et al. (2016)."""

    def __init__(self, rank, world_size, num_classes, overlap):
        super().__init__(rank, world_size, num_classes)
        self.overlap = torch.tensor(overlap, device=self.rank, dtype=torch.float32)

    def __call__(
        self,
        labels,
        predicted):

        tp = torch.zeros(self.num_classes, self.overlap.size(0), device=self.rank, dtype=torch.int64)
        fp = torch.zeros(self.num_classes, self.overlap.size(0), device=self.rank, dtype=torch.int64)

        edges_indices_labels, edges_indices_labels_shifted = self._get_segment_indices(labels, labels.size(1))
        edges_indices_predictions, edges_indices_predictions_shifted = self._get_segment_indices(predicted, predicted.size(1))

        label_segments_used = torch.zeros(edges_indices_labels.size(0), self.overlap.size(0), device=self.rank, dtype=torch.bool)

        # check every segment of predictions for overlap with ground truth
        # segment as a whole is marked as TP/FP/FN, not frame-by-frame
        # earliest correct prediction, for a given ground truth, will be marked TP
        # mark true positive segments (first correctly predicted segment exceeding IoU threshold)
        # mark false positive segments (all further correctly predicted segments exceeding IoU threshold, or those under it)
        # mark false negative segments (all not predicted actual frames)
        for i in range(edges_indices_predictions.size(0)):
            intersection = torch.minimum(edges_indices_predictions_shifted[i], edges_indices_labels_shifted) - torch.maximum(edges_indices_predictions[i], edges_indices_labels)
            union = torch.maximum(edges_indices_predictions_shifted[i], edges_indices_labels_shifted) - torch.minimum(edges_indices_predictions[i], edges_indices_labels)
            # IoU is valid if the predicted class of the segment corresponds to the actual class of the overlapped ground truth segment
            IoU = (intersection/union)*(predicted[0, edges_indices_predictions[i]] == labels[0, edges_indices_labels])
            # ground truth segment with the largest IoU is the (potential) hit
            idx = IoU.argmax()

            # predicted segment is a hit if it exceeds IoU threshold and if its label has not been matched against yet
            hits = torch.bitwise_and(IoU[idx].gt(self.overlap), torch.bitwise_not(label_segments_used[idx]))

            # mark TP and FP correspondingly
            # correctly classified, exceeding the threshold and the first predicted segment to match the ground truth
            tp[predicted[0,edges_indices_predictions[i]]] += hits
            # correctly classified, but under the threshold or not the first predicted segment to match the ground truth
            fp[predicted[0,edges_indices_predictions[i]]] += torch.bitwise_not(hits)
            # mark ground truth segment used if marked TP
            label_segments_used[idx] += hits

        TP = tp.sum(dim=0)
        FP = fp.sum(dim=0)
        # FN are unmatched ground truth segments (misses)
        FN = label_segments_used.size(0) - label_segments_used.sum(dim=0)

        # calculate the F1 score
        precision = TP / (TP+FP)
        recall = TP / (TP+FN)

        self.metric[self.trial_id] = 2*precision*recall/(precision+recall)
        super().__call__()

        return None

    def init_metric(self, num_trials):
        super().init_metric(num_trials)
        self.metric = torch.zeros(self.num_trials, self.overlap.size(0), device=self.rank, dtype=torch.float32)
        return None

    def reduce(self, dst):
        # discard NaN F1 values and compute the macro F1-score (average)
        self.metric = self.metric.nan_to_num(0).mean(dim=0)
        super().reduce(dst)
        self.metric /= self.world_size
        return None

    def save(self, save_dir, suffix):
        pd.DataFrame(torch.stack((self.overlap, self.metric)).cpu().numpy()).to_csv('{0}/macro-F1@k{1}.csv'.format(save_dir, suffix if suffix is not None else ""))
        return None

    def log(self):
        return "f1@k = {0}".format(self.metric.cpu().numpy())
