import torch
import torch.nn.functional as F


class Segment:
    def __init__(self, **kwargs):
        # number of stages
        self.num_stages = kwargs['stages']
        self.num_classes = kwargs['num_classes']
        self.V = kwargs['graph']['num_node']
        self.C = kwargs['in_feat']
        # segment size to divide the trial into (number of predictions to make in a single forward pass)
        self.S = kwargs['segment']
        self.is_ddp = kwargs['is_ddp']
        self.world_size = torch.cuda.device_count()

    def alloc_output(self, L, dtype, rank):
        return torch.zeros(self.stages, self.num_classes, L, dtype=dtype, device=rank)


class BufferSegment(Segment):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # temporal kernel size
        self.G = kwargs['kernel']

    def pad_sequence(self, L):
        # no start padding needed for our RT model because elements are summed internally with a Toeplitz matrix to mimic FIFO behavior
        # NOTE: only needs to overlap the previous segment by the size of the G-1 to mimic prefilled FIFOs to retain same state as if processed continuously
        P_start = 0
        # pad the end to chunk trial into equal size overlapping subsegments to prevent reallocating GPU memory (masks actual outputs later)
        # NOTE: subsegments must overlap by G-1 to continue from the same internal state (first G-1 predictions in subsegments other than first will be discarded)
        temp = (L-self.S)%(self.S-self.G)
        # (temp = 0 - trial splits perfectly, temp < 0 - trial shorter than S, temp > 0 - padding needed to perfectly split)
        P_end = 0 if temp == 0 else (self.S-self.G-temp if temp > 0 else self.S-L)

        # if using DataParallel, make sure the trial is divisible by #GPUs
        num_segments = ((L+P_end-self.S)//(self.S-self.G))+1
        if not self.is_ddp and torch.cuda.is_available():
            temp = num_segments % self.world_size
            P_end += 0 if temp == 0 else (self.world_size-temp)*(self.S-self.G)
        
        return P_start, P_end
    
    def pad_sequence_rt(self, L):
        return 0, 0

    def get_generator(self, L, P_start, P_end):
        if not self.is_ddp and torch.cuda.is_available():
            num_segments = (((L+P_end-self.S)//(self.S-self.G))+1)//self.world_size
            capture_gen = (((self.S-self.G)*(i*self.world_size),self.S+(self.S-self.G)*((i+1)*self.world_size-1)) for i in range(num_segments))
        else:
            num_segments = ((L+P_end-self.S)//(self.S-self.G))+1
            capture_gen = (((self.S-self.G)*i,self.S+(self.S-self.G)*i) for i in range(num_segments))

        return num_segments, capture_gen
    
    def get_generator_rt(self, L):
        return ((i,i+1) for i in range(L))
    
    def get_segment(self, captures, start, end):
        # subsegment of S frames selected (in proposed realtime version)
        data = captures[:,:,start:end]
        
        # if using DataParallel, unfold the resulting chunk into equal length subsegments of user configured segment size
        if not self.is_ddp and torch.cuda.is_available():
            data = data.unfold(2,self.S,self.S-self.G).permute(0,2,1,4,3).contiguous().view(self.world_size,self.C,self.S,self.V)

        return data
    
    def mask_segment(self, L, end, P_start, P_end, i, predictions, labels):
        # if using DataParallel, fold the resulting batched tensor back into single segment 
        if not self.is_ddp and torch.cuda.is_available():
            # clear the overlapping G predictions at the start of each subsegment (except the very first segment)
            predictions[1:,:,:self.G] = 0
            # place the batch size on the last dimension to fold across
            predictions = predictions[None].permute(0,2,3,1).contiguous()
            predictions = predictions.view(1,self.num_classes*self.S,self.world_size)
            
            # fold segments of the original trial computed in parallel on multiple executors back into original length sequence
            predictions = F.fold(
                predictions,
                output_size=(1, self.S+(self.S-self.G)*(self.world_size-1)),
                kernel_size=(1, self.S),
                stride=(1, self.S-self.G))[:,:,0]
            
            # select correponding labels to compare against
            ground_truth = labels[:,self.world_size*(self.S-self.G)*i+(0 if i == 0 else self.G):self.S+(self.S-self.G)*((i+1)*self.world_size-1) if end <= L else L]
        else:
            # select correponding labels to compare against
            ground_truth = labels[:,(self.S-self.G)*i+(0 if i == 0 else self.G):self.S+(self.S-self.G)*i if end <= L else L]

        # clear the overlapping G-1 predictions at the start of each segment (except the very first segment)
        predictions = predictions if i == 0 else predictions[:,:,self.G-1:]
        # drop the outputs corresponding to end-padding (last subsegment)
        predictions = predictions if end <= L else predictions[:,:,:-P_end]
        
        return predictions, ground_truth


class WindowSegment(Segment):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # TODO: provide reduced temporal resolution case
        # window size
        self.W = kwargs['receptive_field']
    
    def pad_sequence(self, L):
        # pad the start by the receptive field size (emulates empty buffer)
        P_start = self.W-1
        # pad the end to chunk trial into equal size subsegments to prevent reallocating GPU memory (masks actual outputs later)
        temp = (L+P_start-(self.S+self.W-1))%(self.S-1)
        P_end = 0 if temp == 0 else (self.S-1-temp if temp > 0 else self.S-L-P_start)

        return P_start, P_end
    
    def pad_sequence_rt(self, L):
        # zero pad the input across time from start by the receptive field size
        # TODO: provide case for different amount of overlap
        return self.W-1, 0

    def get_generator(self, L, P_start, P_end):
        num_segments = ((L+P_start+P_end-(self.S+self.W-1))//(self.S-1))+1
        capture_gen = (((self.S-1)*i,self.W+(self.S-1)*(i+1)) for i in range(num_segments))

        return num_segments, capture_gen
    
    def get_generator_rt(self, L):
        # splits trial for `original` model into overlapping subsequences of samples to separately feed into the model
        return ((i,i+self.W) for i in range(L))
    
    def get_segment(self, captures, start, end):
        # subsegment split into S separate predictions in batch dimension (consequent predictions in original ST-GCN are independent)
        # TODO: change stride of unfolding for temporal resolution reduction
        return captures[:,:,start:end].unfold(2,self.W,1).permute(0,2,1,4,3).contiguous().view(self.S,self.C,self.W,self.V)

    def mask_segment(self, L, end, P_start, P_end, i, predictions, labels):
        # arrange tensor back into a time series
        # (N',C',1)->(N,S,C')
        predictions = predictions.view(1, self.S, self.num_classes)
        # (N,S,C')->(N,C',S)
        predictions = predictions.permute(0, 2, 1)
        # drop the outputs corresponding to end-padding (last subsegment)
        predictions = predictions if end <= (L+P_start) else predictions[:,:,:-P_end]
        # select correponding labels to compare against
        ground_truth = labels[:,(self.S-1)*i+(0 if i == 0 else 1):(self.S-1)*(i+1)+1 if end <= (L+P_start) else L]

        return predictions, ground_truth


class WindowSegmentMultiStage(WindowSegment):
    def mask_segment(self, L, end, P_start, P_end, i, predictions, labels):
        # # arrange tensor back into a time series
        # # (M,N',C',1)->(M,N,S,C')
        # predictions = predictions.view(self.num_stages, 1, self.S, self.num_classes)
        # # (M,N,S,C')->(M,N,C',S)
        # predictions = predictions.permute(0, 1, 3, 2)
        # drop the outputs corresponding to end-padding (last subsegment)
        predictions = predictions if end <= (L+P_start) else predictions[:,:,:,:-P_end]
        # select correponding labels to compare against
        ground_truth = labels[:,(self.S-1)*i+(0 if i == 0 else 1):(self.S-1)*(i+1)+1 if end <= (L+P_start) else L]

        return predictions, ground_truth


class WindowSegmentOneToOneMultiStage(Segment):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # segment size to divide the trial into (number of predictions to make in a single forward pass)
        self.S = kwargs['segment']

    def pad_sequence(self, L):
        P_start = 0
        P_end = self.S-(L%self.S)

        return P_start, P_end
    
    def pad_sequence_rt(self, L):
        raise NotImplementedError('Must implement RT padding for one-to-one models (e.g. MS-TCN).')
    
    def get_generator(self, L, P_start, P_end):
        num_segments = (L+P_end)//self.S
        capture_gen = ((self.S*i,self.S*(i+1)) for i in range(num_segments))

        return num_segments, capture_gen
    
    def get_generator_rt(self, L):
        raise NotImplementedError('Must implement RT generator building for one-to-one models (e.g. MS-TCN).')

    def get_segment(self, captures, start, end):
        return captures[:,:,start:end]
    
    def mask_segment(self, L, end, P_start, P_end, i, predictions, labels):
        # (M,N,C,L)
        predictions = predictions if end <= L else predictions[:,:,:,:-P_end]
        ground_truth = labels[:,self.S*i:self.S*(i+1)]

        return predictions, ground_truth
