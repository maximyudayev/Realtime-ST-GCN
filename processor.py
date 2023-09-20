import math
import torch
from torch.distributed import barrier, reduce
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import pandas as pd
import time
import os
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx


class Processor:
    """ST-GCN processing wrapper for training and testing the model.

    NOTE: if using multi-GPU DDP setup, effective 'emulated' batch size is multiplied by `world_size` (for simpler code).

    TODO: update method descriptions

    Methods:
        train()
            Trains the model, given user-defined training parameters.

        test()
            Performs only the forward pass for inference.
    """

    def __init__(
        self,
        model,
        num_classes,
        class_dist,
        rank):
        """
        Instantiates weighted CE loss and MSE loss.

        Args:
            model : ``torch.nn.Module``
                Configured PyTorch model.

            num_classes : ``int``
                Number of action classification classes.

            dataloader : ``torch.utils.data.DataLoader``
                Data handle to account for its class imbalance in the CE loss.
        """

        self.rank = rank
        self.model = model
        # CE guides model toward absolute correctness on single frame predictions
        self.ce = nn.CrossEntropyLoss(weight=(1-class_dist/torch.sum(class_dist)).to(rank), reduction='mean')
        # MSE component punishes large variations in class probabilities between consecutive samples
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = num_classes


    def model_size(self, save_dir):
        """Returns size of the model."""

        temp_file = "{0}/temp.pt".format(save_dir)
        torch.save({"model_state_dict": self.model.state_dict()}, temp_file)
        size = os.path.getsize(temp_file)/1e6
        os.remove(temp_file)
        return size


    def update_lr_(self, learning_rate, learning_rate_decay, epoch):
        """Decays learning rate monotonically by the provided factor."""

        rate = learning_rate * pow(learning_rate_decay, epoch)
        for g in self.optimizer.param_groups:
            g['lr'] = rate


    def get_segment_indices(self, x, L):
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


    def f1_score(
        self,
        overlap,
        labels,
        predicted):
        """Computes segmental F1@k score with an IoU threshold."""

        tp = torch.zeros(self.num_classes, overlap.size(0), device=self.rank, dtype=torch.int64)
        fp = torch.zeros(self.num_classes, overlap.size(0), device=self.rank, dtype=torch.int64)

        edges_indices_labels, edges_indices_labels_shifted = self.get_segment_indices(labels, labels.size(1))
        edges_indices_predictions, edges_indices_predictions_shifted = self.get_segment_indices(predicted, predicted.size(1))

        label_segments_used = torch.zeros(edges_indices_labels.size(0), overlap.size(0), device=self.rank, dtype=torch.bool)

        # check every segment of predictions for overlap with ground truth
        # segment as a whole is marked as TP/FP/FN, not frame-by-frame
        # earliest correct prediction, for a given ground truth, will be marked TP
        # mark true positive segments (first correctly predicted segment exceeding IoU threshold)
        # mark false positive segments (all further correctly predicted segments exceeding IoU threshold, or those under it)
        # mark false negative segments (all not predicted actual frames)
        for i in range(edges_indices_predictions.size(0)):
            intersection = torch.minimum(edges_indices_predictions_shifted[i],edges_indices_labels_shifted) - torch.maximum(edges_indices_predictions[i],edges_indices_labels)
            union = torch.maximum(edges_indices_predictions_shifted[i],edges_indices_labels_shifted) - torch.minimum(edges_indices_predictions[i],edges_indices_labels)
            # IoU is valid if the predicted class of the segment corresponds to the actual class of the overlapped ground truth segment
            IoU = (intersection/union)*(predicted[0,edges_indices_predictions[i]]==labels[0,edges_indices_labels])
            # ground truth segment with the largest IoU is the (potential) hit
            idx = IoU.argmax()

            # predicted segment is a hit if it exceeds IoU threshold and if its label has not been matched against yet
            hits = torch.bitwise_and(IoU[idx].gt(overlap),torch.bitwise_not(label_segments_used[idx]))

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
        return 2*precision*recall/(precision+recall)


    def edit_score(
        self,
        labels,
        predicted):
        """Computes segmental edit score (Levenshtein distance) between two sequences."""

        edges_indices_labels, _ = self.get_segment_indices(labels, labels.size(1))
        edges_indices_predictions, _ = self.get_segment_indices(predicted, predicted.size(1))

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

        return (1 - D[-1, -1] / max(m_row, n_col))


    def confusion_matrix(
        self,
        labels,
        predicted):
        """Accumulates framewise confusion matrix."""

        confusion = torch.zeros(self.num_classes, self.num_classes, device=self.rank, dtype=torch.int64)
        N, L = labels.size()

        # collect the correct predictions for each class and total per that class
        # for batch_el in range(N*M):
        for batch_el in range(N):
            # OHE 3D matrix, where label and prediction at time `t` are indices
            top1_ohe = torch.zeros(L, self.num_classes, self.num_classes, device=self.rank, dtype=torch.bool)
            top1_ohe[range(L), predicted[batch_el], labels[batch_el]] = True

            # sum-reduce OHE 3D matrix to get number of true vs. false classifications for each class on this sample
            confusion += torch.sum(top1_ohe, dim=0)

        return confusion


    def forward_(
        self,
        captures,
        labels,
        **kwargs):
        """Generator that does the forward pass on the model.

        If `dataset_type` is `'dir'`, processes 1 trial at a time, chops each sequence 
        into equal segments that are split across available executors (GPUs) for parallel computation.

        If `model` is `'original'` and `latency` is `True`, applies the original classification model
        on non-overlapping windows of size `receptive_field` over the input stream, producing outputs at a 
        reduced temporal resolution inversely proportional to the size of the window. Trades prediction
        resolution for compute (does not compute redundant values for input frames otherwise overlapped by 
        multiple windows).

        TODO: automatically split the trial into segments to maximize use of available memory.
        TODO: provide different stride settings for the original model (not only the extremas).
        TODO: add a drawing to the repo to clarify how data segmenting is done.
        """

        # move both data to the compute device
        # (captures is a batch of full-length captures, label is a batch of ground truths)
        # (N,C,L,V)
        captures, labels = captures.to(self.rank), labels.to(self.rank)

        N, C, L, V = captures.size()

        # Splits trial into overlapping subsequences of samples
        if kwargs['dataset_type'] == 'dir':
            if kwargs['model'] == 'original':
                # TODO: provide reduced temporal resolution case for the original model
                # window size
                W = kwargs['receptive_field']
                # segment size to divide the trial into (number of predictions to make in a single forward pass)
                S = kwargs['segment']
                # temporal kernel size
                G = kwargs['kernel'][0]
                # pad the start by the receptive field size (emulates empty buffer)
                P_start = W-1
                # pad the end to chunk trial into equal size subsegments to prevent reallocating GPU memory (masks actual outputs later)
                temp = (L+P_start-(S+W-1))%(S-1)
                P_end = 0 if temp == 0 else (S-1-temp if temp > 0 else S-L-P_start)
            elif kwargs['model'] == 'realtime':
                # segment size to divide the trial into (number of input frames to ingest at a time)
                S = kwargs['segment']
                # temporal kernel size
                G = kwargs['kernel'][0]
                # no start padding needed for our RT model because elements are summed internally with a Toeplitz matrix to mimic FIFO behavior
                # NOTE: only needs to overlap the previous segment by the size of the G-1 to mimic prefilled FIFOs to retain same state as if processed continuously
                P_start = 0
                # pad the end to chunk trial into equal size overlapping subsegments to prevent reallocating GPU memory (masks actual outputs later)
                # NOTE: subsegments must overlap by G-1 to continue from the same internal state (first G-1 predictions in subsegments other than first will be discarded)
                temp = (L-(S-1))%(S-G)
                # (temp = 0 - trial splits perfectly, temp < 0 - trial shorter than S, temp > 0 - padding needed to perfectly split)
                P_end = 0 if temp == 0 else (S-G+1-temp if temp > 0 else S-L)
            else:
                raise NotImplementedError('Not supported model type in `forward_` implementation for directory-based dataset')

            captures = F.pad(captures, (0, 0, P_start, P_end))

            # generator comprehension for lazy processing using start and end indices of subsegments
            if kwargs['model'] == 'original':
                num_segments = ((L+P_start+P_end-(S+W-1))//(S-1))+1
                capture_gen = (((S-1)*i,W+(S-1)*(i+1)) for i in range(num_segments))
            elif kwargs['model'] == 'realtime':
                num_segments = ((L+P_end-S)//(S-G))+1
                capture_gen = (((S-G)*i,S+(S-G)*i) for i in range(num_segments))
            else:
                raise NotImplementedError('Not supported model type in `forward_` implementation for directory-based dataset')

            # generate results for the consumer (effectively limits processing burden by splitting long sequence into manageable independent overlapping chunks)
            for i, (start, end) in enumerate(capture_gen):
                if kwargs['model'] == 'original':
                    # subsegment split into S separate predictions in batch dimension (consequent predictions in original ST-GCN are independent)
                    # TODO: change stride of unfolding for temporal resolution reduction
                    data = captures[:,:,start:end].unfold(2,W,1).permute(0,2,1,4,3).contiguous().view(N*S,C,W,V)
                elif kwargs['model'] == 'realtime':
                    # subsegment of S frames selected (in proposed realtime version)
                    data = captures[:,:,start:end]
                else:
                    raise NotImplementedError('Not supported model type in `forward_` implementation for directory-based dataset')

                # make predictions and compute the loss
                # forward pass the minibatch through the model for the corresponding subject
                # the input tensor has shape (N, C, L, V): N-batch, V-nodes, C-channels, L-length
                # the output tensor has shape (N, C', L)
                predictions = self.model(data)

                if kwargs['model'] == 'original':
                    # arrange tensor back into a time series
                    # (N',C',1)->(N,S,C')
                    predictions = predictions.view(N, S, self.num_classes)
                    # (N,S,C')->(N,C',S)
                    predictions = predictions.permute(0, 2, 1)
                    # drop the outputs corresponding to end-padding (last subsegment)
                    predictions = predictions if end <= (L+P_start) else predictions[:,:,:-P_end]
                    # select correponding labels to compare against
                    ground_truth = labels[:,(S-1)*i+(0 if i == 0 else 1):(S-1)*(i+1)+1 if end <= (L+P_start) else L]
                elif kwargs['model'] == 'realtime':
                    # clear the overlapping G-1 predictions at the start of each segment (except the very first segment)
                    predictions = predictions if i == 0 else predictions[:,:,G:]
                    # drop the outputs corresponding to end-padding (last subsegment)
                    predictions = predictions if end <= L else predictions[:,:,:-P_end]
                    # select correponding labels to compare against
                    ground_truth = labels[:,(S-G)*i+(0 if i == 0 else G):S+(S-G)*i if end <= L else L]
                else:
                    raise NotImplementedError('Not supported model type in `forward_` implementation for directory-based dataset')

                # cross-entropy expects output as class indices (N, C, K), with labels (N, K): 
                # N-batch (flattened multi-skeleton minibatch), C-class, K-extra dimension (segment length)
                # CE + MSE loss metric tuning is taken from @BenjaminFiltjens's MS-GCN:
                # NOTE: subsegments have an overlap of 1 in outputs to enable correct MSE calculation, CE calculation should avoid double counting that frame
                ce = self.ce(predictions[:,:,1 if i!=0 else 0:], ground_truth)
                # in the reduced temporal resolution setting of the original model, MSE loss is expected to be large the higher
                # the receptive field since after that many frames a human could start performing a drastically diferent action
                mse = 0.15 * torch.mean(
                    torch.clamp(
                        self.mse(
                            F.log_softmax(predictions[:,:,1:], dim=1), 
                            F.log_softmax(predictions.detach()[:,:,:-1], dim=1)),
                        min=0,
                        max=16))

                # average the errors across subsegments of a trial
                ce /= num_segments
                mse /= num_segments

                # calculate the predictions statistics
                # this only sums the number of top-1 correctly predicted frames, but doesn't look at prediction jitter
                # NOTE: avoid double counting single overlapping frame occuring in two consequent segments
                top5_probs, top5_predicted = torch.topk(F.softmax(predictions[:,:,1 if i!=0 else 0:], dim=1), k=5, dim=1)
                top1_predicted = top5_predicted[:,0,:]
                # top5_probs[0,:,torch.bitwise_and(torch.any(top5_predicted == labels[:,None,:], dim=1), top1_predicted != labels)[0]].permute(1,0) # probabilities of classes where top-1 and top-5 don't intersect
                top1_cor = torch.sum(top1_predicted == ground_truth).data.item()
                top5_cor = torch.sum(top5_predicted == ground_truth[:,None]).data.item()
                tot = ground_truth.numel()
                # lazy yield statistics for each segment of the trial
                yield top1_predicted, top5_predicted, ground_truth, top1_cor, top5_cor, tot, ce, mse, None
        else:
            raise NotImplementedError('Did not provide a safe `forward_` implementation for file-based dataset types since #93df7ae')


    def forward_rt_(
        self,
        captures,
        labels,
        **kwargs):
        """Generator that does the continual forward pass on the inference-only model."""

        # move both data to the compute device
        # (captures is a batch of full-length captures, label is a batch of ground truths)
        captures, labels = captures.to(self.rank), labels.to(self.rank)

        N, _, L, _ = captures.size()

        latency = 0
        predictions = torch.zeros(N, self.num_classes, L, dtype=captures.dtype, device=self.rank)

        # Splits trial into overlapping subsequences of samples
        if kwargs['dataset_type'] == 'dir': 
            # Splits trial for `original` model into overlapping subsequences of samples to separately feed into the model
            if kwargs['model'] == 'original':
                # zero pad the input across time from start by the receptive field size
                # TODO: provide case for different amount of overlap
                captures = F.pad(captures, (0, 0, kwargs['receptive_field']-1, 0))
                capture_gen = ((i,i+kwargs['receptive_field']) for i in range(L))
            else:
                capture_gen = ((i,i+1) for i in range(L))

            # generate results for the consumer (effectively limits processing burden by splitting long sequence into manageable independent overlapping chunks)
            for i, (start, end) in enumerate(capture_gen):
                start_time = time.time()
                predictions[:,:,i:i+1] = self.model(captures[:,:,start:end])
                latency += (time.time() - start_time)

            # cross-entropy expects output as class indices (N, C, K), with labels (N, K): 
            # N-batch (flattened multi-skeleton minibatch), C-class, K-extra dimension (capture length)
            # CE + MSE loss metric tuning is taken from @BenjaminFiltjens's MS-GCN:
            # CE guides model toward absolute correctness on single frame predictions,
            # MSE component punishes large variations in class probabilities between consecutive samples
            ce = self.ce(predictions, labels)
            # In the reduced temporal resolution setting of the original model, MSE loss is expected to be large the higher
            # the receptive field since after that many frames a human could start performing a drastically diferent action
            mse = 0.15 * torch.mean(
                torch.clamp(
                    self.mse(
                        F.log_softmax(predictions[:,:,1:], dim=1), 
                        F.log_softmax(predictions.detach()[:,:,:-1], dim=1)),
                    min=0,
                    max=16))

            # calculate the predictions statistics
            # this only sums the number of top-1 correctly predicted frames, but doesn't look at prediction jitter
            top5_probs, top5_predicted = torch.topk(F.softmax(predictions, dim=1), k=5, dim=1)
            top1_predicted = top5_predicted[:,0,:]
            # top5_probs[0,:,torch.bitwise_and(torch.any(top5_predicted == labels[:,None,:], dim=1), top1_predicted != labels)[0]].permute(1,0) # probabilities of classes where top-1 and top-5 don't intersect
            top1_cor = torch.sum(top1_predicted == labels).data.item()
            top5_cor = torch.sum(top5_predicted == labels[:,None]).data.item()
            tot = labels.numel()

            yield top1_predicted, top5_predicted, labels, top1_cor, top5_cor, tot, ce, mse, latency/L
        else:
            raise NotImplementedError('Did not provide a safe `forward_rt_` implementation for file-based dataset types since #cc77c393')


    def validate_(
        self,
        dataloader,
        foo,
        num_samples=None,
        **kwargs):
        """Does a forward pass without recording gradients. 

        Shared between train and test scripts: train invokes it after each epoch trained,
        test invokes it once for inference only.
        """

        # do not record gradients
        with torch.no_grad():    
            top1_correct = 0
            top5_correct = 0
            total = 0

            test_start_time = time.time()

            ce_epoch_loss_val = 0
            mse_epoch_loss_val = 0

            latency = 0

            # stats for F1@k segmentation-detection metric of Lea, et al. (2016)
            threshold = torch.tensor(kwargs['iou_threshold'], device=self.rank, dtype=torch.float32)
            f1 = torch.zeros(len(dataloader), threshold.size(0), device=self.rank, dtype=torch.float32)

            # stats for segmental edit score
            edit = torch.zeros(len(dataloader), 1, device=self.rank, dtype=torch.float32)

            # stats for the framewise confusion matrix
            confusion_matrix = torch.zeros(self.num_classes, self.num_classes, device=self.rank, dtype=torch.int64)

            # sweep through the validation dataset in minibatches
            # calculate IoU for the F1@k metrics
            with self.model.join():
                for k, (captures, labels) in enumerate(dataloader):
                    # don't loop through entire dataset - useful to calibrate quantized model or to get the latency metric
                    if k == num_samples: break
                    
                    top1_predicted = []
                    for segment_top1_predicted, _, _, top1_cor, top5_cor, tot, ce, mse, lat in foo(captures, labels, **kwargs):
                        # epoch loss has to multiply by minibatch size to get total non-averaged loss, 
                        # which will then be averaged across the entire dataset size, since
                        # loss for dataset with equal-length trials averages the CE and MSE losses for each minibatch
                        # (used for statistics)
                        ce_epoch_loss_val += ce.data.item()
                        mse_epoch_loss_val += mse.data.item()

                        # evaluate the model
                        top1_correct += top1_cor
                        top5_correct += top5_cor
                        total += tot
                        top1_predicted.append(segment_top1_predicted)
                    
                    latency += lat if lat else 0

                    top1 = torch.concat(top1_predicted, dim=1)
                    labels = labels.to(self.rank)

                    f1[k] = self.f1_score(threshold, labels, top1)
                    edit[k] += self.edit_score(labels, top1)
                    confusion_matrix += self.confusion_matrix(labels, top1)

            test_end_time = time.time()
            duration = test_end_time - test_start_time

            top1_acc = top1_correct / total
            top5_acc = top5_correct / total
            # discard NaN F1 values and compute the macro F1-score (average)
            F1 = f1.nan_to_num(0).mean(dim=0)
            edit = edit.mean(dim=0)

        return top1_acc, top5_acc, F1, edit, confusion_matrix, ce_epoch_loss_val, mse_epoch_loss_val, latency/k if latency else duration


    def train_(
        self,
        dataloader,
        **kwargs):
        """Does one epoch of forward and backward passes on each minibatch in the dataloader."""

        ce_epoch_loss_train = 0
        mse_epoch_loss_train = 0

        top1_correct = 0
        top5_correct = 0
        total = 0

        # sweep through the training dataset in minibatches
        # TODO: make changes for file dataset type
        with self.model.join():
            for i, (captures, labels) in enumerate(dataloader):
                # generator that returns lazy iterator over segments of the trial to process long sequence in manageable overlapping chunks to fit in memory
                for _, _, _, top1_cor, top5_cor, tot, ce, mse, _ in self.forward_(captures, labels, **kwargs):
                    top1_correct += top1_cor
                    top5_correct += top5_cor
                    total += tot

                    # epoch loss has to multiply by minibatch size to get total non-averaged loss, 
                    # which will then be averaged across the entire dataset size, since
                    # loss for dataset with equal-length trials averages the CE and MSE losses for each minibatch
                    # (used for statistics)
                    ce_epoch_loss_train += ce.data.item()
                    mse_epoch_loss_train += mse.data.item()

                    # loss is not a mean across minibatch for different-length trials -> needs averaging
                    loss = ce + mse
                    if (kwargs['dataset_type'] == 'dir' and
                        (len(dataloader) % kwargs['batch_size'] == 0 or
                        i < len(dataloader) - (len(dataloader) % kwargs['batch_size']))):

                        # if the minibatch is the same size as requested (first till one before last minibatch)
                        # (because dataset is a multiple of batch size or if current minibatch is of requested size)
                        loss /= kwargs['batch_size']
                    elif (kwargs['dataset_type'] == 'dir'and 
                        (len(dataloader) % kwargs['batch_size'] != 0 and
                        i >= len(dataloader) - (len(dataloader) % kwargs['batch_size']))):

                        # if the minibatch is smaller than requested (last minibatch or data partition is smaller than requested batch size)
                        loss /= (len(dataloader) % kwargs['batch_size'])

                    print(
                        "[rank {0}, trial {1}]: loss = {2}"
                        .format(self.rank, i, loss),
                        flush=True,
                        file=kwargs['log'][0])

                    # backward pass to compute the gradients
                    # NOTE: after each `backward()`, gradients are synchronized across ranks to ensure same state prior to optimization.
                    # each rank contributes `1/world_size` to the gradient calculation
                    loss.backward()

                # zero the gradient buffers after every batch
                # if dataset is a tensor with equal length trials, always enters
                # if dataset is a set of different length trials, enters every `batch_size` iteration or during last incomplete batch
                if ((kwargs['dataset_type'] == 'dir' and
                        ((i + 1) % kwargs['batch_size'] == 0 or 
                        (i + 1) == len(dataloader))) or
                    (kwargs['dataset_type'] == 'file')):

                    # update parameters based on the computed gradients
                    self.optimizer.step()

                    # clear the gradients
                    self.optimizer.zero_grad()

        return top1_correct, top5_correct, total, ce_epoch_loss_train, mse_epoch_loss_train


    def train(
        self,
        world_size,
        save_dir, 
        train_dataloader,
        val_dataloader,
        epochs,
        checkpoints,
        checkpoint,
        learning_rate,
        learning_rate_decay,
        **kwargs):
        """Trains the model, given user-defined training parameters."""

        # setup the optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # load the checkpoint if not training from scratch
        if checkpoint:
            # TODO: identify/input where the model was trained (i.e. CPU/GPU) and setup map_location automatically
            # NOTE: now assumes model was trained on GPU and maps memory to other distributed GPU processes
            map_location = {'cuda:%d' % 0: 'cuda:%d' % self.rank} if not math.isnan(self.rank) else None
            state = torch.load(checkpoint, map_location=map_location)
            self.optimizer.load_state_dict(state['optimizer_state_dict'])
            range_epochs = range(state['epoch']+1, epochs)
        else:
            range_epochs = range(epochs)

        # variables for email updates
        if not torch.cuda.is_available() or self.rank==0:
            epoch_list = []

            top1_acc_train_list = []
            top1_acc_val_list = []
            top5_acc_train_list = []
            top5_acc_val_list = []
            duration_train_list = []
            duration_val_list = []

            ce_loss_train_list = []
            mse_loss_train_list = []
            epoch_loss_train_list = []

            ce_loss_val_list = []
            mse_loss_val_list = []
            epoch_loss_val_list = []

        # train the model for num_epochs
        # (dataloader is automatically shuffled after each epoch)
        for epoch in range_epochs:
            if torch.cuda.is_available():
                train_dataloader.sampler.set_epoch(epoch)
                val_dataloader.sampler.set_epoch(epoch)

            # set layers to training mode if behavior of any differs between train and prediction
            # (prepares Dropout and BatchNormalization layers to disable and to learn parameters, respectively)
            self.model.train()

            # decay learning rate every 10 epochs [ref: Yan 2018]
            if (epoch % 10 == 0):
                self.update_lr_(learning_rate, learning_rate_decay, epoch//10)

            if self.rank == 0:
                epoch_start_time = time.time()

            # clear the gradients before next epoch
            self.optimizer.zero_grad()

            top1_correct_train, top5_correct_train, total_train, ce_epoch_loss_train, mse_epoch_loss_train = self.train_(train_dataloader, **kwargs)

            print(
                "[rank {0}]: completed training"
                .format(self.rank),
                flush=True,
                file=kwargs['log'][0])

            loss_train = torch.tensor([ce_epoch_loss_train, mse_epoch_loss_train], device=self.rank)
            top1_correct_train = torch.tensor([top1_correct_train], device=self.rank)
            top5_correct_train = torch.tensor([top5_correct_train], device=self.rank)
            total_train = torch.tensor([total_train], device=self.rank)

            if self.rank == 0:
                epoch_end_time = time.time()
                duration_train = epoch_end_time - epoch_start_time
            
            # gather train stats to the master node
            if torch.cuda.is_available():
                reduce(loss_train, dst=0, op=torch.distributed.ReduceOp.SUM)
                reduce(top1_correct_train, dst=0, op=torch.distributed.ReduceOp.SUM)
                reduce(top5_correct_train, dst=0, op=torch.distributed.ReduceOp.SUM)
                reduce(total_train, dst=0, op=torch.distributed.ReduceOp.SUM)

            # checkpoint the model during training at specified epochs
            if (epoch in checkpoints) and ((self.rank == 0 and torch.cuda.is_available()) or not torch.cuda.is_available()):
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.module.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "loss": loss_train.sum() / (len(train_dataloader) * 1 if world_size is None else world_size),
                    }, "{0}/epoch-{1}.pt".format(save_dir, epoch))
            
            # if torch.cuda.is_available():
            #     barrier()

            # set layers to inference mode if behavior differs between train and prediction
            # (prepares Dropout and BatchNormalization layers to enable and to freeze parameters, respectively)
            self.model.eval()

            # test the model on the validation set
            # will complain on CUDA devices that input gradients are none: irrelevant because it is a side effect of
            # the shared `forward_()` routine for both tasks, where the model is set to `train()` or `eval()` in the
            # corresponding caller function
            top1_acc_val, top5_acc_val, f1_score, edit_score, confusion_matrix, ce_epoch_loss_val, mse_epoch_loss_val, duration_val = self.validate_(
                dataloader=val_dataloader,
                foo=self.forward_,
                **kwargs)

            print(
                "[rank {0}]: completed validation"
                .format(self.rank),
                flush=True,
                file=kwargs['log'][0])

            loss_val = torch.tensor([ce_epoch_loss_val, mse_epoch_loss_val], device=self.rank)
            top1_acc_val = torch.tensor([top1_acc_val], device=self.rank)
            top5_acc_val = torch.tensor([top5_acc_val], device=self.rank)

            # gather val stats to the master node
            if torch.cuda.is_available():
                reduce(loss_val, dst=0, op=torch.distributed.ReduceOp.SUM)
                reduce(top1_acc_val, dst=0, op=torch.distributed.ReduceOp.SUM)
                reduce(top5_acc_val, dst=0, op=torch.distributed.ReduceOp.SUM)
                reduce(f1_score, dst=0, op=torch.distributed.ReduceOp.SUM)
                reduce(edit_score, dst=0, op=torch.distributed.ReduceOp.SUM)
                reduce(confusion_matrix, dst=0, op=torch.distributed.ReduceOp.SUM)

                # average values that need to be averaged
                top1_acc_val /= world_size
                top5_acc_val /= world_size
                f1_score /= world_size
                edit_score /= world_size
            
            # record all stats of interest for logging/notification (on master node only if using GPUs)
            if not torch.cuda.is_available() or self.rank==0:
                top1_acc_train = top1_correct_train / total_train
                top5_acc_train = top5_correct_train / total_train
                
                epoch_list.insert(0, epoch)

                ce_loss_train_list.insert(0, (loss_train[0] / (len(train_dataloader) * 1 if world_size is None else world_size)).cpu().item())
                mse_loss_train_list.insert(0, (loss_train[1] / (len(train_dataloader) * 1 if world_size is None else world_size)).cpu().item())
                epoch_loss_train_list.insert(0, (loss_train.sum() / (len(train_dataloader) * 1 if world_size is None else world_size)).cpu().item())

                ce_loss_val_list.insert(0, (loss_val[0] / (len(val_dataloader) * 1 if world_size is None else world_size)).cpu().item())
                mse_loss_val_list.insert(0, (loss_val[1] / (len(val_dataloader) * 1 if world_size is None else world_size)).cpu().item())
                epoch_loss_val_list.insert(0, (loss_val.sum() / (len(val_dataloader) * 1 if world_size is None else world_size)).cpu().item())

                top1_acc_train_list.insert(0, (top1_acc_train).cpu().item())
                top1_acc_val_list.insert(0, (top1_acc_val).cpu().item())
                top5_acc_train_list.insert(0, (top5_acc_train).cpu().item())
                top5_acc_val_list.insert(0, (top5_acc_val).cpu().item())
                duration_train_list.insert(0, duration_train)
                duration_val_list.insert(0, duration_val)
 
                # save all metrics
                pd.DataFrame(torch.stack((torch.tensor(kwargs['iou_threshold'], dtype=torch.float32), f1_score.cpu())).numpy()).to_csv('{0}/macro-F1@k.csv'.format(save_dir))
                pd.DataFrame(data={"top1": top1_acc_val.cpu().numpy(), "top5": top5_acc_val.cpu().numpy()}, index=[0,1]).to_csv('{0}/accuracy.csv'.format(save_dir))
                pd.DataFrame(data={"edit": edit_score.cpu().numpy()}, index=[0]).to_csv('{0}/edit.csv'.format(save_dir))
                pd.DataFrame(confusion_matrix.cpu().numpy()).to_csv('{0}/confusion-matrix.csv'.format(save_dir))

                # sweep through the sample trials
                for i in kwargs["demo"]:
                    # save prediction and ground truth of reference samples
                    captures, labels = val_dataloader.dataset.__getitem__(i)

                    # move both data to the compute device
                    # (captures is a batch of full-length captures, label is a batch of ground truths)
                    captures, labels = captures[None].to(self.rank), labels[None].to(self.rank)

                    top1_predicted = []
                    for segment_top1_predicted, _, _, _, _, _, _, _, _ in self.forward_(captures, labels, **kwargs):
                        top1_predicted.append(segment_top1_predicted)

                    pd.DataFrame(torch.stack((labels[0], torch.concat(top1_predicted, dim=1)[0])).cpu().numpy()).to_csv('{0}/segmentation-{1}.csv'.format(save_dir, i))

                # log and send notifications
                print(
                    "[epoch {0}]: epoch_train_loss = {1}, epoch_val_loss = {2}, top1_acc_train = {3}, top5_acc_train = {4}, top1_acc_val = {5}, top5_acc_val = {6}, f1@k = {7}, edit = {8}"
                    .format(
                        epoch, 
                        (loss_train.sum() / (len(train_dataloader) * 1 if world_size is None else world_size)).cpu().numpy(),
                        (loss_val.sum() / (len(val_dataloader) * 1 if world_size is None else world_size)).cpu().numpy(),
                        top1_acc_train.cpu().numpy(),
                        top5_acc_train.cpu().numpy(),
                        top1_acc_val.cpu().numpy(),
                        top5_acc_val.cpu().numpy(),
                        f1_score.cpu().numpy(),
                        edit_score.cpu().numpy()),
                    flush=True,
                    file=kwargs['log'][0])

                if kwargs['verbose'] > 0:
                    print(
                        "[epoch {0}]: train_time = {1}, val_time = {2}"
                        .format(
                            epoch,
                            duration_train,
                            duration_val),
                        flush=True,
                        file=kwargs['log'][0])

                # format a stats table (in newest to oldest order) and send it as email update
                if kwargs['verbose'] > 1:
                    os.system(
                        'header="\n %-6s %11s %9s %11s %11s %9s %9s %11s %9s\n";'
                        'format=" %-03d %4.6f %4.6f %1.4f %1.4f %1.4f %1.4f %5.6f %5.6f\n";'
                        'printf "$header" "EPOCH" "LOSS_TRAIN" "LOSS_VAL" "TOP1_TRAIN" "TOP5_TRAIN" "TOP1_VAL" "TOP5_VAL" "TIME_TRAIN" "TIME_VAL" > $SLURM_SUBMIT_DIR/mail_draft_{1}.txt;'
                        'printf "$format" {0} >> $SLURM_SUBMIT_DIR/mail_draft_{1}.txt;'
                        'cat $SLURM_SUBMIT_DIR/mail_draft_{1}.txt | mail -s "[{1}]: status update" {2}'
                        .format(
                            ' '.join([
                                ' '.join([str(e) for e in t]) for t 
                                in zip(
                                    epoch_list,
                                    epoch_loss_train_list,
                                    top1_acc_train_list,
                                    top5_acc_train_list,
                                    top1_acc_val_list,
                                    top5_acc_val_list,
                                    duration_train_list,
                                    duration_val_list)]),
                            kwargs['jobname'],
                            kwargs['email']))

                # save (update) train-validation curve as a CSV file after each epoch
                pd.DataFrame(
                    data={
                        'top1_train': top1_acc_train_list,
                        'top1_val': top1_acc_val_list,
                        'top5_train': top5_acc_train_list,
                        'top5_val': top5_acc_val_list
                    }).to_csv('{0}/accuracy-curve.csv'.format(save_dir))

                # save (update) loss curve as a CSV file after each epoch
                pd.DataFrame(
                    data={
                        'ce_train': ce_loss_train_list,
                        'mse_train': mse_loss_train_list,
                        'ce_val': ce_loss_val_list,
                        'mse_val': mse_loss_val_list,
                    }).to_csv('{0}/train-validation-curve.csv'.format(save_dir))

        # save the final model
        if (epoch in checkpoints) and ((self.rank == 0 and torch.cuda.is_available()) or not torch.cuda.is_available()):
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.module.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": loss_train.sum() / (len(train_dataloader) * 1 if world_size is None else world_size),
                }, "{0}/final.pt".format(save_dir))
        # barrier()

        return

    # TODO: update DDP `gather` calls with `reduce`, email notification and file saving
    def test(
        self,
        rank,
        world_size,
        save_dir,
        dataloader,
        **kwargs):
        """Performs only the forward pass for inference."""

        # set layers to inference mode if behavior differs between train and prediction
        # (prepares Dropout and BatchNormalization layers to enable and to freeze parameters, respectively)
        self.model.eval()

        # test the model on the validation set
        top1_acc_val, top5_acc_val, f1_score, edit_score, confusion_matrix, ce_epoch_loss_val, mse_epoch_loss_val, duration_val = self.validate_(
            dataloader=dataloader,
            foo=self.forward_,
            **kwargs)

        # gather val stats to the master node
        if torch.cuda.is_available():
            val_objects = {
                "top1_acc_val": top1_acc_val,
                "top5_acc_val": top5_acc_val,
                "ce_epoch_loss_val": ce_epoch_loss_val,
                "mse_epoch_loss_val": mse_epoch_loss_val,
                "duration_val": duration_val}

            val_output = [None for _ in range(world_size)]
            f1_scores = [torch.zeros(len(kwargs['iou_threshold']), device=self.rank, dtype=torch.float32) for _ in range(world_size)]
            edit_scores = [torch.zeros(1, device=self.rank, dtype=torch.float32) for _ in range(world_size)]
            confusion_matrices = [torch.zeros(self.num_classes, self.num_classes, device=self.rank, dtype=torch.int64) for _ in range(world_size)]

            gather_object(val_objects, val_output if rank == 0 else None, dst=0)
            gather(f1_score, f1_scores if rank == 0 else None, dst=0)
            gather(edit_score, edit_scores if rank == 0 else None, dst=0)
            gather(confusion_matrix, confusion_matrices if rank == 0 else None, dst=0)
        
        if rank == 0:
            for i in range(1, world_size):
                top1_acc_val += val_output[i]["top1_acc_val"]
                top5_acc_val += val_output[i]["top5_acc_val"]
                ce_epoch_loss_val += val_output[i]["ce_epoch_loss_val"]
                mse_epoch_loss_val += val_output[i]["mse_epoch_loss_val"]
                edit_score += edit_scores[i]
                f1_score += f1_scores[i]
                confusion_matrix += confusion_matrices[i]

            top1_acc_val /= world_size
            top5_acc_val /= world_size
            edit_score /= world_size
            f1_score /= world_size
        
        # record all stats of interest for logging/notification (on master node only if using GPUs)
        if not torch.cuda.is_available() or rank==0:
            # save all metrics
            pd.DataFrame(torch.stack((torch.tensor(kwargs['iou_threshold'],dtype=torch.float32),f1_score.cpu())).numpy()).to_csv('{0}/macro-F1@k.csv'.format(save_dir))
            pd.DataFrame(data={"top1": top1_acc_val, "top5": top5_acc_val}, index=[0]).to_csv('{0}/accuracy.csv'.format(save_dir))
            pd.DataFrame(data={"edit": edit_score.cpu().numpy()}, index=[0]).to_csv('{0}/edit.csv'.format(save_dir))
            pd.DataFrame(confusion_matrix.cpu().numpy()).to_csv('{0}/confusion-matrix.csv'.format(save_dir))

            # sweep through the sample trials
            for i in kwargs["demo"]:
                # save prediction and ground truth of reference samples
                captures, labels = dataloader.dataset.__getitem__(i)

                # move both data to the compute device
                # (captures is a batch of full-length captures, label is a batch of ground truths)
                captures, labels = captures[None].to(self.rank), labels[None].to(self.rank)

                top1_predicted = []
                for segment_top1_predicted, _, _, _, _, _, _, _, _ in self.forward_(captures, labels, **kwargs):
                    top1_predicted.append(segment_top1_predicted)

                pd.DataFrame(torch.stack((labels[0], torch.concat(top1_predicted, dim=1)[0])).cpu().numpy()).to_csv('{0}/segmentation-{1}.csv'.format(save_dir, i))

            # log and send notifications
            print(
                "[test]: epoch_loss = {0}, top1_acc = {1}, top5_acc = {2}, f1@k = {3}, edit = {4}"
                .format(
                    (ce_epoch_loss_val + mse_epoch_loss_val) / len(dataloader),
                    top1_acc_val,
                    top5_acc_val,
                    f1_score.cpu().numpy(),
                    edit_score.cpu().numpy()),
                flush=True,
                file=kwargs['log'][0])

            if kwargs['verbose'] > 0:
                print(
                    "[test]: time = {1}"
                    .format(duration_val),
                    flush=True,
                    file=kwargs['log'][0])

            # format a stats table (in newest to oldest order) and send it as email update
            if kwargs['verbose'] > 1:
                os.system(
                    'header="\n %-5s %5s %5s\n";'
                    'format=" %-1.4f %1.4f %5.6f\n";'
                    'printf "$header" "TOP1" "TOP5" "TIME" > $PBS_O_WORKDIR/mail_draft_{1}.txt;'
                    'printf "$format" {0} >> $PBS_O_WORKDIR/mail_draft_{1}.txt;'
                    'cat $PBS_O_WORKDIR/mail_draft_{1}.txt | mail -s "[{1}]: status update" {2}'
                    .format(
                        ' '.join([
                            ' '.join([str(e) for e in t]) for t 
                            in zip(
                                top1_acc_val,
                                top5_acc_val,
                                duration_val)]),
                        '_'.join([kwargs['model'], 'red' if kwargs['model'] == 'original' and kwargs['latency'] else '', *kwargs['jobname']]),
                        kwargs['email']))
        return

    # TODO: complete the benchmark routine
    def benchmark(
        self,
        save_dir,
        dataloader,
        **kwargs):
        """Benchmarks FP32 and INT8 (PTSQ) performance of a model.

        NOTE: currently only supports `original` and `realtime` on `dir` dataset types.
        """

        # replace trainable layers of the proposed model with quantizeable inference-only version
        # TODO: extend to the original ST-GCN
        if kwargs['model'] == 'realtime':
            self.model._swap_layers_for_inference()
            self.model.eval_()
        self.model.eval()
        
        # measure FP32 latency on CPU
        _, _, _, _, _, _, _, latency_fp32 = self.validate_(
            dataloader=dataloader,
            foo=self.forward_rt_,
            device="cpu",
            num_samples=1,
            **kwargs)

        print(
            "[benchmark FP32]: {0} spf"
            .format(latency_fp32),
            flush=True,
            file=kwargs['log'][0])

        size_fp32 = self.model_size(save_dir)

        # prepare the FP32 model: attaches observers to all layers, including the custom layer
        qconfig = torch.ao.quantization.get_default_qconfig_mapping(kwargs['backend'])
        self.model = prepare_fx(self.model, qconfig_mapping=qconfig, example_inputs=torch.randn(1,3,1,25), prepare_custom_config=kwargs['prepare_dict'])

        # move the model to the compute device(s) if available (CPU, GPU, TPU, etc.)
        if torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "allocated GPUs", flush=True, file=kwargs['log'][0])
            self.model = nn.DataParallel(self.model)
        self.model.to(device)
        self.model.eval()
        
        # calibrate the observed model (must be done on the same device as training)
        top1_acc_cal, top5_acc_cal, f1_score_cal, edit_score_cal, confusion_matrix_cal, ce_epoch_loss_cal, mse_epoch_loss_cal, _ = self.validate_(
            dataloader=dataloader,
            foo=self.forward_rt_,
            device=device,
            num_samples=1,
            **kwargs)

        # # sweep through the sample trials
        # for i in [175, 293, 37]: # 39 (L=4718), 177 (L=6973), 299 (L=2378)
        #     # save prediction and ground truth of reference samples
        #     captures, labels = dataloader.dataset.__getitem__(i)

        #     # move both data to the compute device
        #     # (captures is a batch of full-length captures, label is a batch of ground truths)
        #     captures, labels = captures[None].to(device), labels[None].to(device)

        #     top1_predicted = []
        #     for segment_top1_predicted, _, _, _, _, _, _, _, _ in self.forward_rt_(captures, labels, device=device, **kwargs):
        #         top1_predicted.append(segment_top1_predicted)

        #     pd.DataFrame(torch.stack((labels[0], torch.concat(top1_predicted, dim=1)[0])).cpu().numpy()).to_csv('{0}/segmentation-{1}_fp32.csv'.format(save_dir, i))

        # move the model to the CPU for INT8 inference
        self.model.to("cpu")

        # quantize the calibrated model
        self.model = convert_fx(self.model, qconfig_mapping=qconfig, convert_custom_config=kwargs['convert_dict'])
        self.model.eval()

        self.model = torch.jit.script(self.model)

        # evaluate performance of the PTSQ model (must be done on the same device as training)
        # measure INT8 latency
        top1_acc_q, top5_acc_q, f1_score_q, edit_score_q, confusion_matrix_q, ce_epoch_loss_q, mse_epoch_loss_q, latency_int8 = self.validate_(
            dataloader=dataloader,
            foo=self.forward_rt_,
            device="cpu",
            num_samples=1,
            **kwargs)

        print(
            "[benchmark INT8]: {0} spf"
            .format(latency_int8),
            flush=True,
            file=kwargs['log'][0])

        size_int8 = self.model_size(save_dir)

        # # sweep through the sample trials
        # for i in [175, 293, 37]: # 39 (L=4718), 177 (L=6973), 299 (L=2378)
        #     # save prediction and ground truth of reference samples
        #     captures, labels = dataloader.dataset.__getitem__(i)

        #     # move both data to the compute device
        #     # (captures is a batch of full-length captures, label is a batch of ground truths)
        #     captures, labels = captures[None].to("cpu"), labels[None].to("cpu")

        #     top1_predicted = []
        #     for segment_top1_predicted, _, _, _, _, _, _, _, _ in self.forward_rt_(captures, labels, device="cpu", **kwargs):
        #         top1_predicted.append(segment_top1_predicted)

        #     pd.DataFrame(torch.stack((labels[0], torch.concat(top1_predicted, dim=1)[0])).cpu().numpy()).to_csv('{0}/segmentation-{1}_int8.csv'.format(save_dir, i))

        # save all the measurements
        pd.DataFrame(
            data={
                'top1_fp32': top1_acc_cal,
                'top1_int8': top1_acc_q,
                'top5_fp32': top5_acc_cal,
                'top5_int8': top5_acc_q
            },
            index=[0]).to_csv('{0}/accuracy.csv'.format(save_dir))

        pd.DataFrame(
            data={
                'ce_fp32': ce_epoch_loss_cal,
                'mse_fp32': mse_epoch_loss_cal,
                'ce_int8': ce_epoch_loss_q,
                'mse_int8': mse_epoch_loss_q,
            },
            index=[0]).to_csv('{0}/loss.csv'.format(save_dir))

        pd.DataFrame(torch.stack((torch.tensor(kwargs['iou_threshold'],dtype=torch.float32),f1_score_cal.cpu(),f1_score_q.cpu())).numpy()).to_csv('{0}/macro-F1@k.csv'.format(save_dir))
        pd.DataFrame(data={"edit_fp32": edit_score_cal.cpu().numpy(), "edit_int8": edit_score_q.cpu().numpy()},index=[0]).to_csv('{0}/edit.csv'.format(save_dir))
        pd.DataFrame(data={"latency_fp32": latency_fp32, "latency_int8": latency_int8},index=[0]).to_csv('{0}/latency.csv'.format(save_dir))
        pd.DataFrame(data={"size_fp32": size_fp32, "size_int8": size_int8},index=[0]).to_csv('{0}/model-size.csv'.format(save_dir))
        pd.DataFrame(confusion_matrix_cal.cpu().numpy()).to_csv('{0}/confusion-matrix_fp32.csv'.format(save_dir))
        pd.DataFrame(confusion_matrix_q.cpu().numpy()).to_csv('{0}/confusion-matrix_int8.csv'.format(save_dir))

        return
