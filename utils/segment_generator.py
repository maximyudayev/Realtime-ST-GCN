import torch
import torch.nn.functional as F


class Segment:
    def __init__(self, rank, world_size, **kwargs):
        # number of stages
        self.num_stages = kwargs['stages']
        self.num_classes = kwargs['num_classes']
        self.V = kwargs['graph']['num_node']
        self.C = kwargs['in_feat']
        self.rank = rank
        self.world_size = world_size

    def alloc_output(self, L, dtype):
        return torch.zeros(self.stages, self.num_classes, L, dtype=dtype, device=self.rank)


class BufferSegment(Segment):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # temporal kernel size
        self.G = kwargs['kernel']

    def pad_sequence(self, L):
        # no start padding needed for our RT model because elements are summed internally with a Toeplitz matrix to mimic FIFO behavior
        # NOTE: only needs to overlap the previous segment by the size of the G-1 to mimic prefilled FIFOs to retain same state as if processed continuously
        P_start = 0

        # NOTE: subsegments must overlap by G-1 to continue from the same internal state (first G-1 predictions in subsegments other than first will be discarded)
        temp = (L-(self.world_size-1)*(self.G-1))%self.world_size
        # (temp = 0 - trial splits perfectly, temp < 0 - trial shorter than S, temp > 0 - padding needed to perfectly split)
        P_end = 0 if temp == 0 else (self.world_size-temp)
        
        self.S = ((L+P_end-(self.world_size-1)*(self.G-1))//self.world_size)+(0 if self.world_size==1 else self.G-1)

        return P_start, P_end

    def get_segment(self, captures):
        return captures.unfold(2,self.S,self.S-self.G).permute(0,2,1,4,3).contiguous().view(self.world_size,self.C,self.S,self.V)

    def mask_segment(self, L, P_start, P_end, predictions):
        # clear the overlapping G predictions at the start of each subsegment (except the very first segment)
        predictions[1:,:,:self.G] = 0
        # place the batch size on the last dimension to fold across
        predictions = predictions[None].permute(0,2,3,1).contiguous()
        predictions = predictions.view(1,self.num_classes*self.S,self.world_size)

        # fold segments of the original trial computed in parallel on multiple executors back into original length sequence
        # output_size=(1, self.S+(self.S-self.G)*(self.world_size-1)),
        predictions = F.fold(
            predictions,
            output_size=(1, L+P_end),
            kernel_size=(1, self.S),
            stride=(1, self.S-self.G))[:,:,0]

        return predictions[:,:,:L]


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
        # arrange tensor back into a time series
        # (M,N',C',1)->(M,N,L,C')->(M,N,C',L)
        return predictions.view(self.num_stages, 1, self.S, self.num_classes).permute(0,1,3,2)


class WindowSegmentOneToOneMultiStage(Segment):
    # train on 1 GPU
    def pad_sequence(self, L):
        return 0, 0

    def get_segment(self, captures):
        return captures
    
    def mask_segment(self, L, P_start, P_end, predictions):
        # (M,N,C,L)
        return predictions
