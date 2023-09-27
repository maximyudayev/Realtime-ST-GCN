import torch
import pandas as pd
from torch.distributed import reduce

class Metric:
    """Base class for metrics to capture."""

    def __init__(self, rank, world_size, num_classes):
        self.num_classes = num_classes
        self.rank = rank
        self.world_size = world_size

    def __call__(self):
        self.trial_id += 1
        return None

    def _get_segment_indices(self, x, L):
        """Detects edges in a sequence.

        Will yield arbitrary non-zero values at the edges of classes."""

        edges = torch.zeros(L, device=self.rank, dtype=torch.int64)
        edges[0] = 1
        edges[1:] = x[0,1:]-x[0,:-1]

        edges_indices = edges.nonzero()[:,0]
        edges_indices_shifted = torch.zeros_like(edges_indices, device=self.rank, dtype=torch.int64)
        edges_indices_shifted[:-1] = edges_indices[1:]
        edges_indices_shifted[-1] = L

        return edges_indices, edges_indices_shifted

    def init_metric(self, num_trials):
        self.num_trials = num_trials
        self.trial_id = 0
        return None

    def value(self):
        return self.metric

    def reduce(self, dst):
        reduce(self.metric, dst=dst, op=torch.distributed.ReduceOp.SUM)
        return None

    def save(self, save_dir, suffix):
        raise NotImplementedError('Must override the `_save` method implementation for the custom metric.')

    def log(self):
        raise NotImplementedError('Must override the `_log` method implementation for the custom metric.')


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


class EditScore(Metric):
    """Computes segmental edit score (Levenshtein distance) between two sequences."""

    def __call__(
        self,
        labels,
        predicted):

        edges_indices_labels, _ = self._get_segment_indices(labels, labels.size(1))
        edges_indices_predictions, _ = self._get_segment_indices(predicted, predicted.size(1))

        # collect the segmental edit score
        m_row = edges_indices_predictions.size(0)
        n_col = edges_indices_labels.size(0)

        D = torch.zeros(m_row+1, n_col+1, device=self.rank, dtype=torch.float32)
        D[:,0] = torch.arange(m_row+1)
        D[0,:] = torch.arange(n_col+1)

        for j in range(1, n_col+1):
            for i in range(1, m_row+1):
                if labels[0,edges_indices_labels][j-1] == predicted[0,edges_indices_predictions][i-1]:
                    D[i, j] = D[i - 1, j - 1]
                else:
                    D[i, j] = min(D[i - 1, j] + 1,
                                D[i, j - 1] + 1,
                                D[i - 1, j - 1] + 1)

        self.metric[self.trial_id] = (1 - D[-1, -1] / max(m_row, n_col))
        super().__call__()

        return None

    def init_metric(self, num_trials):
        super().init_metric(num_trials)
        self.metric = torch.zeros(self.num_trials, 1, device=self.rank, dtype=torch.float32)
        return None

    def reduce(self, dst):
        self.metric = self.metric.mean(dim=0)
        super().reduce(dst)
        self.metric /= self.world_size
        return None

    def save(self, save_dir, suffix):
        pd.DataFrame(data={"edit": self.metric.cpu().numpy()}, index=[0]).to_csv('{0}/edit{1}.csv'.format(save_dir, suffix if suffix is not None else ""))
        return None

    def log(self):
        return "edit = {0}".format(self.metric.cpu().numpy())


class ConfusionMatrix(Metric):
    """Accumulates framewise confusion matrix."""

    def __call__(
        self,
        labels,
        predicted):

        N, L = labels.size()

        # collect the correct predictions for each class and total per that class
        # for batch_el in range(N*M):
        for batch_el in range(N):
            # OHE 3D matrix, where label and prediction at time `t` are indices
            top1_ohe = torch.zeros(L, self.num_classes, self.num_classes, device=self.rank, dtype=torch.bool)
            top1_ohe[range(L), predicted[batch_el], labels[batch_el]] = True

            # sum-reduce OHE 3D matrix to get number of true vs. false classifications for each class on this sample
            self.metric += torch.sum(top1_ohe, dim=0)

        return None

    def init_metric(self, num_trials):
        super().init_metric(num_trials)
        self.metric = torch.zeros(self.num_classes, self.num_classes, device=self.rank, dtype=torch.int64)
        return None

    def save(self, save_dir, suffix):
        pd.DataFrame(self.metric.cpu().numpy()).to_csv('{0}/confusion-matrix{1}.csv'.format(save_dir, suffix if suffix is not None else ""))
        return None

    def log(self):
        return None
