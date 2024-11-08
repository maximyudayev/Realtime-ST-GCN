import torch


class Metric:
    """Base class for metrics to capture."""

    def __init__(self, rank, num_classes):
        self.num_classes = num_classes
        self.rank = rank

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

    def reduce(self):
        return None

    def save(self, save_dir, suffix):
        raise NotImplementedError('Must override the `_save` method implementation for the custom metric.')

    def log(self):
        raise NotImplementedError('Must override the `_log` method implementation for the custom metric.')
