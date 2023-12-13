import torch
from .metric import Metric
import pandas as pd


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
