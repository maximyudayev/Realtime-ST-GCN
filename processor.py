import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, reduce, broadcast_object_list, broadcast, barrier

from torch.utils.data import DataLoader
from data_prep.dataset import SkeletonDataset, SkeletonDatasetFromDirectory

from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

from contextlib import nullcontext
import pandas as pd
import json
import time
import os


def _build_model(Model, rank: int, args):
    """Builds the selected ST-GCN model variant.

    Args:
        args : ``dict``
            Parsed CLI arguments.

    Returns:
        PyTorch Model corresponding to the user-defined CLI parameters, configured as DDP if using GPUs.
    """

    model = Model(rank, **vars(args))

    # if using GPUs, convert the created model into the DistributedDataParallel
    if torch.cuda.is_available():
        model = DDP(model.to(rank), device_ids=[rank])

    # load the checkpoint if not trained from scratch
    if args.checkpoint:
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank} if not (rank is None) else None
        state = torch.load(args.checkpoint, map_location=map_location)
        model.module.load_state_dict(state['model_state_dict'])
    if torch.cuda.is_available(): barrier()

    return model


def _build_dataloader(rank, world_size, args):
    """Builds dataloaders that supply data in the format (N,C,L,V).

    TODO: use pin_memory to load each sample directly in the corresponding GPU.
    """

    # preparing datasets for training and validation
    if args.dataset_type == 'file':
        train_data = SkeletonDataset('{0}/train_data.npy'.format(args.data), '{0}/train_label.pkl'.format(args.data), args.actions)
        val_data = SkeletonDataset('{0}/val_data.npy'.format(args.data), '{0}/val_label.pkl'.format(args.data), args.actions)
    elif args.dataset_type == 'dir':
        train_data = SkeletonDatasetFromDirectory('{0}/train/features'.format(args.data), '{0}/train/labels'.format(args.data), args.actions)
        val_data = SkeletonDatasetFromDirectory('{0}/val/features'.format(args.data), '{0}/val/labels'.format(args.data), args.actions)
    else:
        raise NotImplementedError('Unsupported dataset type in `setup`. Currently supports `file` and `dir`.')

    # trials of different length can not be placed in the same tensor when batching, have to manually iterate over them
    batch_size = 1 if args.dataset_type == 'dir' else args.batch_size

    # prepare the DataLoaders
    if torch.cuda.is_available():
        # randomly splits dataset on each epoch into equal number of trials per rank (not repeating indeces), dropping remainder trials
        # each epoch, the sets are shuffled and effectively all trials are used throughout training
        train_sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        val_sampler = DistributedSampler(val_data, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)

        # TODO: use pin_memory to load each sample directly in the corresponding GPU
        train_dataloader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)
        val_dataloader = DataLoader(val_data, batch_size=batch_size, sampler=val_sampler)
    else:
        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    return train_dataloader, val_dataloader


def _get_action_classes(args):
    """Extract actions from the action list file."""

    with open(args.actions, 'r') as action_names:
        actions = action_names.read().split('\n')

    # 0th class is always background action
    action_dict = dict()
    for i, action in enumerate(actions):
        action_dict[i+1] = action

    return action_dict


def _get_class_dist(dataloader, rank):
    # gets distribution of the complete dataset, not just the random sampled subset of DistributedSampler
    return dataloader.dataset.__get_distribution__(rank)


def _get_skeleton_graph(args):
    # extract skeleton graph data
    with open(args.graph, 'r') as graph_file:
        return json.load(graph_file)


def _makedirs(args):
    # retrieve the job name
    jobname = os.getenv('SLURM_JOB_NAME') if os.getenv('SLURM_ARRAY_TASK_ID') is None else os.getenv('SLURM_JOB_NAME')+'_'+os.getenv('SLURM_ARRAY_TASK_ID')

    # prepare a directory to store results
    save_dir = "{0}/{1}".format(
        args.out,
        jobname)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # prepare a backup directory
    if args.backup:
        backup_dir = "{0}/{1}".format(
            args.backup,
            jobname)

        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)
    else:
        backup_dir = None

    return jobname, save_dir, backup_dir


def setup(Model, rank: int, world_size: int, args):
    """Performs setup common to any ST-GCN model variant.

    Only needs to be invoked once for a given problem (train-test, benchmark, etc.). 
    Corresponds to the parts of the pipeline irrespective of the black-box model used.
    Creates DataLoaders, sets up processing device and random number generator,
    reads action classes file.

    TODO: adapt for Torchrun and RPC for multi-node multi-GPU training
    TODO: use pin_memory to load data straight into corresponding GPU

    Args:
        args : ``dict``
            Parsed CLI arguments.

    Returns:
        Dictionary of action classes.

        Train and validation DataLoaders.
    """

    # if using CUDA, initialize process group for DDP using environment variables from the SLURM job script
    if torch.cuda.is_available():
        init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

    # on multi-GPU DDP setup, adjust the learning rate by the number of GPUs to counteract the reduced step of the optimizer
    if torch.cuda.is_available():
        args.learning_rate *= world_size

    # construct dataloaders
    train_dataloader, val_dataloader = _build_dataloader(rank, world_size, args)

    # get action classes
    # NOTE: must be repeated by each process to create proper output tensor size for class distribution
    action_dict = _get_action_classes(args)

    # do some setup only on Master Node and scatter results to other GPUs
    if rank == 0 or not torch.cuda.is_available():
        graph = _get_skeleton_graph(args)

        jobname, save_dir, backup_dir = _makedirs(args)

        # create object list for scattering across processes
        objects = [{
            "graph": graph,
            "save_dir": save_dir,
            "backup_dir": backup_dir,
            "jobname": jobname
        }]

        # get class distribution
        class_dist = _get_class_dist(train_dataloader, rank)
    elif rank != 0 and torch.cuda.is_available():
        objects = [None]
        class_dist = torch.zeros(len(action_dict), dtype=torch.float32, device=rank)

    # scatter results to all other GPUs
    if torch.cuda.is_available():
        broadcast_object_list(objects, src=0)
        broadcast(class_dist, src=0)

    args.num_classes = len(action_dict)
    args.graph, args.save_dir, args.backup_dir, args.jobname = objects[0].values()

    # construct the target model using the user's CLI arguments and automatically convert the model to DDP if using GPUs
    model = _build_model(Model, rank, args)

    return model, train_dataloader, val_dataloader, class_dist, args


def cleanup():
    if torch.cuda.is_available(): destroy_process_group()
    return None


class Processor:
    """ST-GCN processing wrapper for training and testing the model.

    NOTE: if using multi-GPU DDP setup, effective 'emulated' batch size is multiplied by `world_size` (for simpler code).

    TODO: update method descriptions
    TODO: add a drawing to the repo to clarify how data segmenting is done.
    TODO: automatically split the trial into segments to maximize use of available memory.

    Methods:
        train()
            Trains the model, given user-defined training parameters.

        test()
            Performs only the forward pass for inference.

        benchmark()
            Quantizes a model and compares KPIs between the INT8 and FP32 versions.
    """

    def __init__(
        self,
        model,
        metrics,
        num_classes,
        class_dist,
        rank,
        world_size):
        """
        Instantiates weighted CE loss and MSE loss.

        Args:
            model : ``torch.nn.Module``
                Configured PyTorch model.

            metrics : [``Metric``]
                List of metric classes to instantiate and use.

            num_classes : ``int``
                Number of action classification classes.

            class_dist : ``torch.Tensor``
                Total number of class occurences in the dataset to weigh the CE loss.

            rank : ``int``
                Local GPU rank.

            rank : ``int``
                Number of total used GPUs.
        """

        self.rank = rank
        self.world_size = world_size
        self.model = model
        self.metrics = metrics
        # CE guides model toward absolute correctness on single frame predictions
        self.ce = nn.CrossEntropyLoss(weight=(1-class_dist/torch.sum(class_dist)).to(rank), reduction='mean')
        # self.ce = nn.CrossEntropyLoss(reduction='mean')
        # MSE component punishes large variations in class probabilities between consecutive samples
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = num_classes


    def _model_size(self, save_dir):
        """Returns size of the model."""

        temp_file = "{0}/temp.pt".format(save_dir)
        torch.save({"model_state_dict": self.model.module.state_dict()}, temp_file)
        size = os.path.getsize(temp_file)/1e6
        os.remove(temp_file)
        return size


    def _update_lr(self, learning_rate, learning_rate_decay, epoch):
        """Decays learning rate monotonically by the provided factor.

        TODO: replace with a PyTorch rate scheduler.
        """

        rate = learning_rate * pow(learning_rate_decay, epoch)
        for g in self.optimizer.param_groups:
            g['lr'] = rate
        return None


    def _init_metrics(self, num_trials):
        for metric in self.metrics:
            metric.init_metric(num_trials)
        return None


    def _collect_metrics(self, labels, predictions):
        for metric in self.metrics:
            metric(labels, predictions)
        return None


    def _reduce_metrics(self, dst):
        for metric in self.metrics:
            metric.reduce(dst)
        return None


    def _save_metrics(self, save_dir, suffix):
        for metric in self.metrics:
            metric.save(save_dir, suffix)
        return None


    def _log_metrics(self):
        log_list = []
        for metric in self.metrics:
            log = metric.log()
            if log is not None:
                log_list.append(log)
        if len(log_list):
            return ", ".join(["", *log_list])
        else:
            return ""


    def _demo_segmentation_masks(self, dataloader, suffix, demo, save_dir, **kwargs):
        with torch.no_grad():
            # sweep through the sample trials
            for i in demo:
                # save prediction and ground truth of reference samples
                captures, labels = dataloader.dataset.__getitem__(i)

                # move both data to the compute device
                # (captures is a batch of full-length captures, label is a batch of ground truths)
                captures, labels = captures[None].to(self.rank), labels[None].to(self.rank)

                top1_predicted = []
                for segment_top1_predicted, _, _, _, _, _, _, _, _ in self._forward(captures, labels, **kwargs):
                    top1_predicted.append(segment_top1_predicted)

                pd.DataFrame(torch.stack((labels[0], torch.concat(top1_predicted, dim=1)[0])).cpu().numpy()).to_csv('{0}/segmentation-{1}{2}.csv'.format(save_dir, i, suffix if suffix is not None else ""))
        return None


    def _forward(
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

        TODO: provide different stride settings for the original model (not only the extremas).
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
                temp = (L-S)%(S-G)
                # (temp = 0 - trial splits perfectly, temp < 0 - trial shorter than S, temp > 0 - padding needed to perfectly split)
                P_end = 0 if temp == 0 else (S-G-temp if temp > 0 else S-L)
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
                # NOTE: for distributed evaluation with DDP, must forward-pass the model using the local model by calling .module() 
                # to avoid processes hanging in anticipation of synchronization (https://github.com/pytorch/pytorch/issues/54059#issuecomment-801197198)
                predictions = self.model(data) if not (not self.model.training and torch.cuda.is_available()) else self.model.module(data)

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
                    predictions = predictions if i == 0 else predictions[:,:,G-1:]
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


    def _forward_rt(
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


    def _test(
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

            # clear and initialize user-defined metrics for the epoch
            self._init_metrics(len(dataloader))

            # NOTE: forces early finished GPUs to wait for laggards to prevent hanging during load imbalance
            with self.model.join() if torch.cuda.is_available() else nullcontext():

                # sweep through the validation dataset in minibatches
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

                        print(
                            "[rank {0}, trial {1}]: loss = {2}"
                            .format(self.rank, k, ce+mse),
                            flush=True,
                            file=kwargs['log'][0])

                    latency += lat if lat else 0

                    top1 = torch.concat(top1_predicted, dim=1)
                    labels = labels.to(self.rank)

                    # collect user-defined evaluation metrics
                    self._collect_metrics(labels, top1)

                test_end_time = time.time()
                duration = test_end_time - test_start_time

                top1_acc = top1_correct / total
                top5_acc = top5_correct / total

        return top1_acc, top5_acc, ce_epoch_loss_val, mse_epoch_loss_val, latency/k if latency else duration


    def _train(
        self,
        dataloader,
        **kwargs):
        """Does one epoch of forward and backward passes on each minibatch in the dataloader.

        TODO: make changes for file dataset type.
        """

        ce_epoch_loss_train = 0
        mse_epoch_loss_train = 0

        top1_correct = 0
        top5_correct = 0
        total = 0

        # sweep through the training dataset in minibatches
        # NOTE: forces early finished GPUs to wait for laggards to prevent hanging during load imbalance
        with self.model.join() if torch.cuda.is_available() else nullcontext():
            for i, (captures, labels) in enumerate(dataloader):
                # TODO: add model.no_sync() context manager to prevent gradient synchronization until the optimization step
                # generator that returns lazy iterator over segments of the trial to process long sequence in manageable overlapping chunks to fit in memory
                for _, _, _, top1_cor, top5_cor, tot, ce, mse, _ in self._forward(captures, labels, **kwargs):
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

        # load the optimizer checkpoint if not training from scratch
        if checkpoint:
            # TODO: identify/input where the model was trained (i.e. CPU/GPU) and setup map_location automatically
            # NOTE: now assumes model was trained on GPU and maps memory to other distributed GPU processes
            map_location = {'cuda:%d' % 0: 'cuda:%d' % self.rank} if not (self.rank is None) else None
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

        if self.rank == 0 or not torch.cuda.is_available():
            start_time = time.time()
            print("Training started", flush=True, file=kwargs["log"][0])

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
                self._update_lr(learning_rate, learning_rate_decay, epoch//10)

            if self.rank == 0:
                epoch_start_time = time.time()

            # clear the gradients before next epoch
            self.optimizer.zero_grad()

            top1_correct_train, top5_correct_train, total_train, ce_epoch_loss_train, mse_epoch_loss_train = self._train(train_dataloader, **kwargs)

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
                    "loss": loss_train.sum() / (len(train_dataloader) * (1 if self.world_size is None else self.world_size)),
                    }, "{0}/epoch-{1}.pt".format(save_dir, epoch))

            # set layers to inference mode if behavior differs between train and prediction
            # (prepares Dropout and BatchNormalization layers to enable and to freeze parameters, respectively)
            self.model.eval()

            # test the model on the validation set
            # will complain on CUDA devices that input gradients are none: irrelevant because it is a side effect of
            # the shared `forward_()` routine for both tasks, where the model is set to `train()` or `eval()` in the
            # corresponding caller function
            top1_acc_val, top5_acc_val, ce_epoch_loss_val, mse_epoch_loss_val, duration_val = self._test(
                dataloader=val_dataloader,
                foo=self._forward,
                **kwargs)

            print(
                "[rank {0}]: completed testing"
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
                self._reduce_metrics(dst=0)

                # average values that need to be averaged
                top1_acc_val /= self.world_size
                top5_acc_val /= self.world_size

            # record all stats of interest for logging/notification (on master node only if using GPUs)
            if not torch.cuda.is_available() or self.rank==0:
                top1_acc_train = top1_correct_train / total_train
                top5_acc_train = top5_correct_train / total_train

                epoch_list.insert(0, epoch)

                ce_loss_train_list.insert(0, (loss_train[0] / (len(train_dataloader) * (1 if self.world_size is None else self.world_size))).cpu().item())
                mse_loss_train_list.insert(0, (loss_train[1] / (len(train_dataloader) * (1 if self.world_size is None else self.world_size))).cpu().item())
                epoch_loss_train_list.insert(0, (loss_train.sum() / (len(train_dataloader) * (1 if self.world_size is None else self.world_size))).cpu().item())

                ce_loss_val_list.insert(0, (loss_val[0] / (len(val_dataloader) * (1 if self.world_size is None else self.world_size))).cpu().item())
                mse_loss_val_list.insert(0, (loss_val[1] / (len(val_dataloader) * (1 if self.world_size is None else self.world_size))).cpu().item())
                epoch_loss_val_list.insert(0, (loss_val.sum() / (len(val_dataloader) * (1 if self.world_size is None else self.world_size))).cpu().item())

                top1_acc_train_list.insert(0, (top1_acc_train).cpu().item())
                top1_acc_val_list.insert(0, (top1_acc_val).cpu().item())
                top5_acc_train_list.insert(0, (top5_acc_train).cpu().item())
                top5_acc_val_list.insert(0, (top5_acc_val).cpu().item())
                duration_train_list.insert(0, duration_train)
                duration_val_list.insert(0, duration_val)

                # save all metrics
                pd.DataFrame(data={"top1": top1_acc_val.cpu().numpy(), "top5": top5_acc_val.cpu().numpy()}, index=[0,1]).to_csv('{0}/accuracy.csv'.format(save_dir))

                self._save_metrics(save_dir, None)

                self._demo_segmentation_masks(dataloader=val_dataloader, suffix=None, save_dir=save_dir, **kwargs)

                # log and send notifications
                print(
                    "[epoch {0}]: epoch_train_loss = {1}, epoch_val_loss = {2}, top1_acc_train = {3}, top5_acc_train = {4}, top1_acc_val = {5}, top5_acc_val = {6}{7}"
                    .format(
                        epoch,
                        (loss_train.sum() / (len(train_dataloader) * (1 if self.world_size is None else self.world_size))).cpu().numpy(),
                        (loss_val.sum() / (len(val_dataloader) * (1 if self.world_size is None else self.world_size))).cpu().numpy(),
                        top1_acc_train.cpu().numpy(),
                        top5_acc_train.cpu().numpy(),
                        top1_acc_val.cpu().numpy(),
                        top5_acc_val.cpu().numpy(),
                        self._log_metrics()),
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
        if ((self.rank == 0 and torch.cuda.is_available()) or not torch.cuda.is_available()):
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.module.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": loss_train.sum() / (len(train_dataloader) * (1 if self.world_size is None else self.world_size)),
                }, "{0}/final.pt".format(save_dir))

        if self.rank == 0 or not torch.cuda.is_available():
            print("Training completed in: {0}".format(time.time() - start_time), flush=True, file=kwargs["log"][0])

        return None


    def test(
        self,
        save_dir,
        dataloader,
        **kwargs):
        """Performs only the forward pass for inference."""

        if self.rank == 0 or not torch.cuda.is_available():
            start_time = time.time()
            print("Testing started", flush=True, file=kwargs["log"][0])

        # set layers to inference mode if behavior differs between train and prediction
        # (prepares Dropout and BatchNormalization layers to enable and to freeze parameters, respectively)
        self.model.eval()

        # test the model on the test set
        top1_acc_val, top5_acc_val, ce_epoch_loss_val, mse_epoch_loss_val, duration_val = self._test(
            dataloader=dataloader,
            foo=self._forward,
            **kwargs)

        print(
            "[rank {0}]: completed testing"
            .format(self.rank),
            flush=True,
            file=kwargs['log'][0])

        loss_val = torch.tensor([ce_epoch_loss_val, mse_epoch_loss_val], device=self.rank)
        top1_acc_val = torch.tensor([top1_acc_val], device=self.rank)
        top5_acc_val = torch.tensor([top5_acc_val], device=self.rank)

        # gather stats to the master node
        if torch.cuda.is_available():
            reduce(loss_val, dst=0, op=torch.distributed.ReduceOp.SUM)
            reduce(top1_acc_val, dst=0, op=torch.distributed.ReduceOp.SUM)
            reduce(top5_acc_val, dst=0, op=torch.distributed.ReduceOp.SUM)
            self._reduce_metrics(dst=0)

            # average values that need to be averaged
            top1_acc_val /= self.world_size
            top5_acc_val /= self.world_size

        # record all stats of interest for logging/notification (on master node only if using GPUs)
        if not torch.cuda.is_available() or self.rank==0:
            # save all metrics
            pd.DataFrame(data={"top1": top1_acc_val.cpu().numpy(), "top5": top5_acc_val.cpu().numpy()}, index=[0,1]).to_csv('{0}/accuracy.csv'.format(save_dir))

            self._save_metrics(save_dir, None)

            self._demo_segmentation_masks(dataloader=dataloader, suffix=None, save_dir=save_dir, **kwargs)

            # log and send notifications
            print(
                "[test]: epoch_loss = {0}, top1_acc = {1}, top5_acc = {2}{3}"
                .format(
                    (loss_val.sum() / (len(dataloader) * (1 if self.world_size is None else self.world_size))).cpu().numpy(),
                    top1_acc_val.cpu().numpy(),
                    top5_acc_val.cpu().numpy(),
                    self._log_metrics()),
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
                    'printf "$header" "TOP1" "TOP5" "TIME" > $SLURM_SUBMIT_DIR/mail_draft_{1}.txt;'
                    'printf "$format" {0} >> $SLURM_SUBMIT_DIR/mail_draft_{1}.txt;'
                    'cat $SLURM_SUBMIT_DIR/mail_draft_{1}.txt | mail -s "[{1}]: status update" {2}'
                    .format(
                        ' '.join([
                            ' '.join([str(e) for e in t]) for t
                            in zip(
                                top1_acc_val,
                                top5_acc_val,
                                duration_val)]),
                        kwargs['jobname'],
                        kwargs['email']))

        if self.rank == 0 or not torch.cuda.is_available():
            print("Testing completed in: {0}".format(time.time() - start_time), flush=True, file=kwargs["log"][0])

        return None


    # TODO: finish porting to DDP
    def benchmark(
        self,
        save_dir,
        dataloader,
        **kwargs):
        """Benchmarks FP32 and INT8 (PTSQ) performance of a model.

        NOTE: currently only supports `original` and `realtime` on `dir` dataset types.
        """

        if self.rank == 0 or not torch.cuda.is_available():
            start_time = time.time()
            print("Benchmarking started", flush=True, file=kwargs["log"][0])

        # replace trainable layers of the proposed model with quantizeable inference-only version
        # TODO: extend to the ST-GCN
        if kwargs['model'] == 'realtime':
            self.model._swap_layers_for_inference()
            self.model.eval_()
        self.model.eval()
        
        # measure FP32 latency on CPU
        _, _, _, _, latency_fp32 = self._test(
            dataloader=dataloader,
            foo=self._forward_rt,
            num_samples=1,
            **kwargs)

        print(
            "[benchmark FP32]: {0} spf"
            .format(latency_fp32),
            flush=True,
            file=kwargs['log'][0])

        size_fp32 = self._model_size(save_dir)

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
        top1_acc_cal, top5_acc_cal, ce_epoch_loss_cal, mse_epoch_loss_cal, _ = self._test(
            dataloader=dataloader,
            foo=self._forward_rt,
            num_samples=1,
            **kwargs)

        self._save_metrics(save_dir, "_fp32")

        self._demo_segmentation_masks(kwargs["demo"], dataloader, save_dir, "_fp32")

        # move the model to the CPU for INT8 inference
        self.model.to("cpu")

        # quantize the calibrated model
        self.model = convert_fx(self.model, qconfig_mapping=qconfig, convert_custom_config=kwargs['convert_dict'])
        self.model.eval()

        self.model = torch.jit.script(self.model)

        # evaluate performance of the PTSQ model (must be done on the same device as training)
        # measure INT8 latency
        top1_acc_q, top5_acc_q, ce_epoch_loss_q, mse_epoch_loss_q, latency_int8 = self._test(
            dataloader=dataloader,
            foo=self._forward_rt,
            num_samples=1,
            **kwargs)

        print(
            "[benchmark INT8]: {0} spf"
            .format(latency_int8),
            flush=True,
            file=kwargs['log'][0])

        size_int8 = self._model_size(save_dir)

        self._save_metrics(save_dir, "_int8")

        self._demo_segmentation_masks(kwargs["demo"], dataloader, save_dir, "_int8")

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

        pd.DataFrame(data={"latency_fp32": latency_fp32, "latency_int8": latency_int8},index=[0]).to_csv('{0}/latency.csv'.format(save_dir))
        pd.DataFrame(data={"size_fp32": size_fp32, "size_int8": size_int8},index=[0]).to_csv('{0}/model-size.csv'.format(save_dir))

        if self.rank == 0 or not torch.cuda.is_available():
            print("Benchmarking completed in: {0}".format(time.time() - start_time), flush=True, file=kwargs["log"][0])

        return None
