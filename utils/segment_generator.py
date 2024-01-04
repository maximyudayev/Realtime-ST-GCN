import torch
import torch.nn.functional as F


class Segment:
    def __init__(self, output_device, **kwargs):
        # number of stages
        self.num_stages = kwargs['stages']
        self.num_classes = kwargs['num_classes']
        self.V = kwargs['graph']['num_node']
        self.C = kwargs['in_feat']
        self.output_device = output_device

    def alloc_output(self, L, dtype):
        return torch.zeros(self.num_stages, self.num_classes, L, dtype=dtype, device=self.output_device)


class BufferSegment(Segment):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # temporal kernel size
        self.G = kwargs['kernel']

    def pad_sequence(self, L):
        return 0, 0

    def get_segment(self, captures):
        return captures

    def mask_segment(self, L, P_start, P_end, predictions):
        return predictions


class WindowSegment(Segment):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # TODO: provide reduced temporal resolution case
        # window size
        self.W = kwargs['receptive_field']
    
    def pad_sequence(self, L):
        # pad the start by the receptive field size (emulates empty buffer)
        P_start = self.W-1
        P_end = 0
        self.S = L

        return P_start, P_end
    
    def get_segment(self, captures):
        # TODO: change stride of unfolding for temporal resolution reduction
        return captures.unfold(2, self.W, 1).permute(0,2,1,4,3).contiguous().view(self.S, self.C, self.W, self.V)

    def mask_segment(self, L, P_start, P_end, predictions):
        # arrange tensor back into a time series
        # (N',C',1)->(N,L,C')->(N,C',L)
        return predictions.view(1, self.S, self.num_classes).permute(0,2,1)


class WindowSegmentMultiStage(WindowSegment):
    def mask_segment(self, L, P_start, P_end, predictions):
        # (M,N,C,L)
        return predictions


class WindowSegmentOneToOneMultiStage(Segment):
    # train on 1 GPU
    def pad_sequence(self, L):
        return 0, 0

    def get_segment(self, captures):
        return captures
    
    def mask_segment(self, L, P_start, P_end, predictions):
        # (M,N,C,L)
        return predictions
