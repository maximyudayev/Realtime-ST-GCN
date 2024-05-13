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
        return torch.zeros(self.num_stages, self.num_classes, L, dtype=dtype, device=self.rank)


class BufferSegment(Segment):
    # For RT models, "segment" parameter controls subsegment size to fit in the available GPU
    # -1: no sub-segmenting, splits evenly across GPUs
    # X: splits trial into subsegments of this length (intended for single GPU experiments with insufficient memory)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # temporal kernel size
        self.G = kwargs['kernel']
        self.subsegment_size = kwargs['segment']

    def pad_sequence(self, L):
        # no start padding needed for our RT model because elements are summed internally with a Toeplitz matrix to mimic FIFO behavior
        P_start = 0
        self.L = L

        # NOTE: only needs to overlap the previous segment by the size of the G-1 to mimic prefilled FIFOs to retain same state as if processed continuously
        if self.subsegment_size > 0:
            # Splits into defined length subsegments
            # NOTE: subsegments must overlap by G-1 to continue from the same internal state (first G-1 predictions in subsegments other than first will be discarded)
            temp = (L-self.subsegment_size)%(self.subsegment_size-self.G)

            # (temp = 0 - trial splits perfectly, temp < 0 - trial shorter than S, temp > 0 - padding needed to perfectly split)
            P_end = 0 if temp == 0 else (self.subsegment_size-self.G-temp)

            self.P_end = P_end
        else:
            # Splits evenly across available GPUs
            # NOTE: subsegments must overlap by G-1 to continue from the same internal state (first G-1 predictions in subsegments other than first will be discarded)
            temp = (L-(self.world_size-1)*(self.G-1))%self.world_size
            # (temp = 0 - trial splits perfectly, temp < 0 - trial shorter than S, temp > 0 - padding needed to perfectly split)
            P_end = 0 if temp == 0 else (self.world_size-temp)

            self.S = ((L+P_end-(self.world_size-1)*(self.G-1))//self.world_size)+(0 if self.world_size==1 else self.G-1)

        return P_start, P_end

    def pad_sequence_rt(self, L):
        self.L = L
        return 0, 0

    def get_segment(self, captures, labels):
        if self.subsegment_size > 0:
            num_segments = ((self.L+self.P_end-self.subsegment_size)//(self.subsegment_size-self.G))+1
            data = captures.unfold(2,self.subsegment_size,self.subsegment_size-self.G).permute(0,2,1,4,3).contiguous().view(num_segments,self.C,self.subsegment_size,self.V)

            for i in range(num_segments):
                yield \
                    data[i:i+1], \
                    labels[:,0 if i==0 else self.subsegment_size+(self.subsegment_size-self.G)*(i-1):self.subsegment_size+(self.subsegment_size-self.G)*i if i < num_segments-1 else self.L], \
                    num_segments
        else:
            return \
                captures.unfold(2,self.S,self.S-self.G).permute(0,2,1,4,3).contiguous().view(self.world_size,self.C,self.S,self.V), \
                labels, \
                1

    def get_segment_rt(self, captures):
        for i in range(self.L):
            yield captures[:,:,i:i+1]

    def mask_segment(self, i, num_segments, L, P_start, P_end, predictions):
        if self.subsegment_size > 0:
            if i == 0:
                return predictions
            elif i > 0 and i<(num_segments-1):
                return predictions[:,:,self.G-1:]
            else:
                return predictions[:,:,self.G-1:-P_end]
        else:
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
        self.subsegment_size = kwargs['segment']

    def pad_sequence(self, L):
        # pad the start by the receptive field size (emulates empty buffer)
        P_start = self.W-1
        P_end = 0
        self.S = L

        return P_start, P_end

    def pad_sequence_rt(self, L):
        # pad the start by the receptive field size (emulates empty buffer)
        P_start = self.W-1
        P_end = 0
        self.L = L

        return P_start, P_end

    def get_segment(self, captures, labels):
        num_segments = (self.S+self.S%self.subsegment_size)//self.subsegment_size
        segments = ((
                self.subsegment_size*i - (1 if i>0 else 0),
                self.subsegment_size*(i+1)+(self.W-1) if i<num_segments-1 else captures.size(2),
                self.subsegment_size*i,
                self.subsegment_size*(i+1) if i<num_segments-1 else labels.size(1)
                ) for i in range(num_segments))

        for startX, endX, startY, endY in segments:
            yield \
                captures[:,:,startX:endX].unfold(2, self.W, 1).permute(0,2,1,4,3).contiguous().view(endX-startX-(self.W-1), self.C, self.W, self.V), \
                labels[:,startY:endY], \
                num_segments

    def get_segment_rt(self, captures):
        for i in range(self.L):
            yield captures[:,:,i:i+self.W]

    def mask_segment(self, L, P_start, P_end, predictions):
        # arrange tensor back into a time series
        # (N',C',1)->(N,L,C')->(N,C',L)
        return predictions.permute(2,1,0)


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
