import torch
from .metric import Metric
import pandas as pd


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

    def reduce(self):
        self.metric = self.metric.mean(dim=0)
        return None

    def save(self, save_dir, suffix):
        pd.DataFrame(data={"edit": self.metric.cpu().numpy()}, index=[0]).to_csv('{0}/edit{1}.csv'.format(save_dir, suffix if suffix is not None else ""))
        return None

    def log(self):
        return "edit = {0}".format(self.metric.cpu().numpy())
