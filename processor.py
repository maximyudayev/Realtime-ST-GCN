import torch
import torch.nn.functional as F
from torch import optim

from torch.nn import DataParallel as DP

from torch.utils.data import DataLoader
from data_prep import SkeletonDataset, SkeletonDatasetFromDirectory

from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

import pandas as pd
import json
import time
import os


def _build_model(Model, rank, args):
    """Builds the selected ST-GCN model variant.

    Args:
        args : ``dict``
            Parsed CLI arguments.

    Returns:
        PyTorch Model corresponding to the user-defined CLI parameters, configured as DDP if using GPUs.
    """

    model = Model(rank, **args.arch)

    if torch.cuda.device_count() > 1:
        model = DP(model)
    model.to(rank)

    # load the checkpoint if not trained from scratch
    if args.processor.get('checkpoint'):        
        state = torch.load(args.processor['checkpoint'], map_location=rank)
        model.module.load_state_dict(state['model_state_dict'])

    return model


def _build_dataloader(args):
    """Builds dataloaders that supply data in the format (N,C,L,V).

    TODO: use pin_memory to load each sample directly in the corresponding GPU.
    """

    # preparing datasets for training and validation
    if args.processor['dataset_type'] == 'file':
        train_data = SkeletonDataset('{0}/train_data.npy'.format(args.processor['data']), '{0}/train_label.pkl'.format(args.processor['data']), args.processor['actions'])
        val_data = SkeletonDataset('{0}/val_data.npy'.format(args.data), '{0}/val_label.pkl'.format(args.data), args.actions)
    elif args.processor['dataset_type'] == 'dir':
        train_data = SkeletonDatasetFromDirectory('{0}/train/features'.format(args.processor['data']), '{0}/train/labels'.format(args.processor['data']), args.processor['actions'])
        val_data = SkeletonDatasetFromDirectory('{0}/val/features'.format(args.processor['data']), '{0}/val/labels'.format(args.processor['data']), args.processor['actions'])
    else:
        raise NotImplementedError('Unsupported dataset type in `setup`. Currently supports `file` and `dir`.')

    # trials of different length can not be placed in the same tensor when batching, have to manually iterate over them
    batch_size = 1 if args.processor['dataset_type'] == 'dir' else args.optimizer['batch_size']

    # prepare the DataLoaders
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    return train_dataloader, val_dataloader


def _get_action_classes(file):
    """Extract actions from the action list file."""

    with open(file, 'r') as action_names:
        actions = action_names.read().split('\n')

    # 0th class is always background action
    action_dict = dict()
    for i, action in enumerate(actions):
        action_dict[i+1] = action

    return action_dict


def _get_class_dist(dataloader, rank):
    # gets distribution of the complete dataset, not just the random sampled subset of DistributedSampler
    return dataloader.dataset.__get_distribution__(rank)


def _get_skeleton_graph(file):
    # extract skeleton graph data
    with open(file, 'r') as graph_file:
        return json.load(graph_file)


def _makedirs(args):
    # retrieve the job name
    jobname = os.getenv('SLURM_JOB_NAME') if os.getenv('SLURM_ARRAY_TASK_ID') is None else os.getenv('SLURM_JOB_NAME')+'_'+os.getenv('SLURM_ARRAY_TASK_ID')

    # prepare a directory to store results
    save_dir = "{0}/{1}".format(
        args.processor['out'],
        jobname)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # prepare a backup directory
    if args.processor.get('backup'):
        backup_dir = "{0}/{1}".format(
            args.processor['backup'],
            jobname)

        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)
    else:
        backup_dir = None

    return jobname, save_dir, backup_dir


def setup(Model, Loss, SegmentGenerator, Statistics, rank, world_size, args):
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

    # construct dataloaders
    train_dataloader, val_dataloader = _build_dataloader(args)

    # get action classes
    action_dict = _get_action_classes(args.processor['actions'])
    args.arch['num_classes'] = len(action_dict)

    args.arch['graph'] = _get_skeleton_graph(args.processor['graph'])

    args.job['jobname'], args.processor['save_dir'], args.processor['backup_dir'] = _makedirs(args)

    # get class distribution
    train_class_dist = _get_class_dist(train_dataloader, rank)
    val_class_dist = _get_class_dist(val_dataloader, rank)

    # construct the target model using the user's CLI arguments
    model = _build_model(Model, rank, args)
    
    loss = Loss(rank, train_class_dist, args.arch['output_type'])
    segment_generator = SegmentGenerator(rank=rank, world_size=world_size, **args.arch)
    statistics = Statistics()

    return model, loss, segment_generator, statistics, train_dataloader, val_dataloader, args


def cleanup(args):
    return None


class Processor:
    """ST-GCN processing wrapper for training and testing the model.

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
        rank,
        world_size,
        model,
        loss,
        statistics,
        segment_generator,
        metrics):
        """
        Instantiates weighted CE loss and MSE loss.

        Args:
            rank : ``int``
                Local GPU rank.

            world_size : ``int``
                Number of total used GPUs.

            model : ``torch.nn.Module``
                Configured PyTorch model.

            loss : ``Loss``
                Loss object to be used by the processor on the model.

            statistics : ``Statistics``
                Callable class to collect framewise statistics.

            segment_generator : ``Segment``
                Processor helper to chop data entries into overlapping segments, according to the model type.

            metrics : [``Metric``]
                List of metric class instances to collect during processing.
        """

        self.rank = rank
        self.world_size = world_size
        self.model = model
        self.loss = loss
        self.statistics = statistics
        self.segment_generator = segment_generator
        self.metrics = metrics


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


    def _reduce_metrics(self):
        for metric in self.metrics:
            metric.reduce()
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


    def _demo_segmentation_masks(self, dataloader, suffix, demo, save_dir):
        with torch.no_grad():
            # sweep through the sample trials
            for i in demo:
                # save prediction and ground truth of reference samples
                captures, labels = dataloader.dataset.__getitem__(i)

                # move both data to the compute device
                # (captures is a batch of full-length captures, label is a batch of ground truths)
                captures, labels = captures[None].to(self.rank), labels[None].to(self.rank)

                top1_predicted, _, _, _, _, _, _, _, _ = self._forward(captures, labels)

                pd.DataFrame(torch.stack((labels[0], top1_predicted[0])).cpu().numpy()).to_csv('{0}/segmentation-{1}{2}.csv'.format(save_dir, i, suffix if suffix is not None else ""))
        return None


    def _forward(
        self,
        captures,
        labels):
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

        _, _, L, _ = captures.size()

        # pad the sequence according to the model type
        P_start, P_end = self.segment_generator.pad_sequence(L)

        captures = F.pad(captures, (0, 0, P_start, P_end))

        # get the next segment corresponding to the model type and segmentation strategy
        data = self.segment_generator.get_segment(captures)

        # the input tensor has shape (N, C, L, V): N-batch, V-nodes, C-channels, L-length
        # the output tensor has shape (N, C', L)
        predictions = self.model(data)

        # recombine results back into a time-series, corresponding to the segmentation strategy
        predictions = self.segment_generator.mask_segment(L, P_start, P_end, predictions)

        # get the loss of the model
        ce, mse = self.loss(predictions, labels)

        # calculate the predictions statistics
        top1_predicted, top5_predicted, top1_cor, top5_cor, tot = self.statistics(predictions, labels)

        # lazy yield statistics for each segment of the trial
        return top1_predicted, top5_predicted, labels, top1_cor, top5_cor, tot, ce, mse, None


    def _forward_rt(
        self,
        captures,
        labels):
        """Generator that does the continual forward pass on the inference-only model."""

        # move both data to the compute device
        # (captures is a batch of full-length captures, label is a batch of ground truths)
        captures, labels = captures.to(self.rank), labels.to(self.rank)

        _, _, L, _ = captures.size()

        latency = 0
        predictions = self.segment_generator.alloc_output(L, dtype=captures.dtype, device=self.rank)

        P_start, P_end = self.segment_generator.pad_sequence_rt(L)
        
        captures = F.pad(captures, (0, 0, P_start, P_end))

        # generator comprehension for lazy processing using start and end indices of subsegments
        capture_gen = self.segment_generator.get_generator_rt(L)

        # generate results for the consumer (effectively limits processing burden by splitting long sequence into manageable independent overlapping chunks)
        for i, (start, end) in enumerate(capture_gen):
            start_time = time.time()
            predictions[:,:,i:i+1] = self.model(captures[:,:,start:end])
            latency += (time.time() - start_time)

        # get the loss of the model
        ce, mse = self.loss(predictions, labels)

        # calculate the predictions statistics
        top1_predicted, top5_predicted, top1_cor, top5_cor, tot = self.statistics(predictions, labels)

        return top1_predicted, top5_predicted, labels, top1_cor, top5_cor, tot, ce, mse, latency/L


    def _test(
        self,
        dataloader,
        foo,
        log,
        num_samples=None):
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

            # sweep through the validation dataset in minibatches
            for k, (captures, labels) in enumerate(dataloader):
                # don't loop through entire dataset - useful to calibrate quantized model or to get the latency metric
                if k == num_samples: break

                top1_predicted, _, _, top1_cor, top5_cor, tot, ce, mse, lat = foo(captures, labels)
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

                print(
                    "[rank {0}, trial {1}]: loss = {2}"
                    .format(self.rank, k, ce+mse),
                    flush=True,
                    file=log[0])

                latency += lat if lat else 0

                # collect user-defined evaluation metrics
                self._collect_metrics(labels.to(self.rank), top1_predicted)

                test_end_time = time.time()
                duration = test_end_time - test_start_time

                top1_acc = top1_correct / total
                top5_acc = top5_correct / total

        return top1_acc, top5_acc, ce_epoch_loss_val, mse_epoch_loss_val, latency/k if latency else duration


    def _train(
        self,
        dataloader,
        batch_size,
        log):
        """Does one epoch of forward and backward passes on each minibatch in the dataloader.

        TODO: make changes for file dataset type.
        """

        ce_epoch_loss_train = 0
        mse_epoch_loss_train = 0

        top1_correct = 0
        top5_correct = 0
        total = 0

        # sweep through the training dataset in minibatches
        for i, (captures, labels) in enumerate(dataloader):
            _, _, _, top1_cor, top5_cor, tot, ce, mse, _ = self._forward(captures, labels)
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
            if (len(dataloader) % batch_size == 0 or
                i < len(dataloader) - (len(dataloader) % batch_size)):

                # if the minibatch is the same size as requested (first till one before last minibatch)
                # (because dataset is a multiple of batch size or if current minibatch is of requested size)
                loss /= batch_size
            elif (len(dataloader) % batch_size != 0 and
                i >= len(dataloader) - (len(dataloader) % batch_size)):

                # if the minibatch is smaller than requested (last minibatch or data partition is smaller than requested batch size)
                loss /= (len(dataloader) % batch_size)

            print(
                "[rank {0}, trial {1}]: loss = {2}"
                .format(self.rank, i, loss),
                flush=True,
                file=log[0])

            # backward pass to compute the gradients
            # NOTE: after each `backward()`, gradients are synchronized across ranks to ensure same state prior to optimization.
            # each rank contributes `1/world_size` to the gradient calculation
            loss.backward()

            # zero the gradient buffers after every batch
            # if dataset is a tensor with equal length trials, always enters
            # if dataset is a set of different length trials, enters every `batch_size` iteration or during last incomplete batch
            if ((i + 1) % batch_size == 0 or
                (i + 1) == len(dataloader)):

                # update parameters based on the computed gradients
                self.optimizer.step()

                # clear the gradients
                self.optimizer.zero_grad()

        return top1_correct, top5_correct, total, ce_epoch_loss_train, mse_epoch_loss_train


    def train(
        self,
        train_dataloader,
        val_dataloader,
        proc_conf,
        optim_conf,
        job_conf):
        """Trains the model, given user-defined training parameters."""

        # setup the optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=optim_conf['learning_rate'])

        # load the optimizer checkpoint if not training from scratch
        if proc_conf.get('checkpoint'):
            state = torch.load(proc_conf['checkpoint'], map_location=self.rank)
            self.optimizer.load_state_dict(state['optimizer_state_dict'])
            range_epochs = range(state['epoch']+1, optim_conf['epochs'])
        else:
            range_epochs = range(optim_conf['epochs'])

        # variables for email updates
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
        
        start_time = time.time()
        print("Training started", flush=True, file=job_conf["log"][0])

        # train the model for num_epochs
        # (dataloader is automatically shuffled after each epoch)
        for epoch in range_epochs:

            # set layers to training mode if behavior of any differs between train and prediction
            # (prepares Dropout and BatchNormalization layers to disable and to learn parameters, respectively)
            self.model.train()

            # decay learning rate every 10 epochs [ref: Yan 2018]
            if (epoch % 10 == 0):
                self._update_lr(optim_conf['learning_rate'], optim_conf['learning_rate_decay'], epoch//10)

            epoch_start_time = time.time()

            # clear the gradients before next epoch
            self.optimizer.zero_grad()

            top1_correct_train, top5_correct_train, total_train, ce_epoch_loss_train, mse_epoch_loss_train = self._train(
                dataloader=train_dataloader,
                batch_size=optim_conf['batch_size'],
                log=job_conf['log'])

            print(
                "[rank {0}]: completed training"
                .format(self.rank),
                flush=True,
                file=job_conf['log'][0])

            loss_train = torch.tensor([ce_epoch_loss_train, mse_epoch_loss_train], device=self.rank)
            top1_correct_train = torch.tensor([top1_correct_train], device=self.rank)
            top5_correct_train = torch.tensor([top5_correct_train], device=self.rank)
            total_train = torch.tensor([total_train], device=self.rank)

            epoch_end_time = time.time()
            duration_train = epoch_end_time - epoch_start_time

            # checkpoint the model during training at specified epochs
            if (epoch in optim_conf['checkpoint_indices']):
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.module.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "loss": loss_train.sum() / len(train_dataloader),
                    }, "{0}/epoch-{1}.pt".format(proc_conf['save_dir'], epoch))

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
                log=job_conf['log'])

            print(
                "[rank {0}]: completed testing"
                .format(self.rank),
                flush=True,
                file=job_conf['log'][0])
            
            loss_val = torch.tensor([ce_epoch_loss_val, mse_epoch_loss_val], device=self.rank)
            top1_acc_val = torch.tensor([top1_acc_val], device=self.rank)
            top5_acc_val = torch.tensor([top5_acc_val], device=self.rank)

            # reduce metrics and gather to master if using DDP
            self._reduce_metrics()

            top1_acc_train = top1_correct_train / total_train
            top5_acc_train = top5_correct_train / total_train

            epoch_list.insert(0, epoch)

            ce_loss_train_list.insert(0, (loss_train[0] / len(train_dataloader)).cpu().item())
            mse_loss_train_list.insert(0, (loss_train[1] / len(train_dataloader)).cpu().item())
            epoch_loss_train_list.insert(0, (loss_train.sum() / len(train_dataloader)).cpu().item())

            ce_loss_val_list.insert(0, (loss_val[0] / len(val_dataloader)).cpu().item())
            mse_loss_val_list.insert(0, (loss_val[1] / len(val_dataloader)).cpu().item())
            epoch_loss_val_list.insert(0, (loss_val.sum() / len(val_dataloader)).cpu().item())

            top1_acc_train_list.insert(0, (top1_acc_train).cpu().item())
            top1_acc_val_list.insert(0, (top1_acc_val).cpu().item())
            top5_acc_train_list.insert(0, (top5_acc_train).cpu().item())
            top5_acc_val_list.insert(0, (top5_acc_val).cpu().item())
            duration_train_list.insert(0, duration_train)
            duration_val_list.insert(0, duration_val)

            # save all metrics
            pd.DataFrame(data={"top1": top1_acc_val.cpu().numpy(), "top5": top5_acc_val.cpu().numpy()}, index=[0,1]).to_csv('{0}/accuracy.csv'.format(proc_conf['save_dir']))

            self._save_metrics(proc_conf['save_dir'], None)

            self._demo_segmentation_masks(
                dataloader=val_dataloader, 
                suffix=None,
                demo=proc_conf['demo'],
                save_dir=proc_conf['save_dir'])

            # log and send notifications
            print(
                "[epoch {0}]: epoch_train_loss = {1}, epoch_val_loss = {2}, top1_acc_train = {3}, top5_acc_train = {4}, top1_acc_val = {5}, top5_acc_val = {6}{7}"
                .format(
                    epoch,
                    (loss_train.sum() / len(train_dataloader)).cpu().numpy(),
                    (loss_val.sum() / len(val_dataloader)).cpu().numpy(),
                    top1_acc_train.cpu().numpy(),
                    top5_acc_train.cpu().numpy(),
                    top1_acc_val.cpu().numpy(),
                    top5_acc_val.cpu().numpy(),
                    self._log_metrics()),
                flush=True,
                file=job_conf['log'][0])

            if job_conf['verbose'] > 0:
                print(
                    "[epoch {0}]: train_time = {1}, val_time = {2}"
                    .format(
                        epoch,
                        duration_train,
                        duration_val),
                    flush=True,
                    file=job_conf['log'][0])

            # format a stats table (in newest to oldest order) and send it as email update
            if job_conf['verbose'] > 1:
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
                        job_conf['jobname'],
                        job_conf['email']))

            # save (update) train-validation curve as a CSV file after each epoch
            pd.DataFrame(
                data={
                    'top1_train': top1_acc_train_list,
                    'top1_val': top1_acc_val_list,
                    'top5_train': top5_acc_train_list,
                    'top5_val': top5_acc_val_list
                }).to_csv('{0}/accuracy-curve.csv'.format(proc_conf['save_dir']))

            # save (update) loss curve as a CSV file after each epoch
            pd.DataFrame(
                data={
                    'ce_train': ce_loss_train_list,
                    'mse_train': mse_loss_train_list,
                    'ce_val': ce_loss_val_list,
                    'mse_val': mse_loss_val_list,
                }).to_csv('{0}/train-validation-curve.csv'.format(proc_conf['save_dir']))

        # save the final model
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.module.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": loss_train.sum() / len(train_dataloader),
            }, "{0}/final.pt".format(proc_conf['save_dir']))

        print("Training completed in: {0}".format(time.time() - start_time), flush=True, file=job_conf["log"][0])

        return None


    def test(
        self,
        dataloader,
        proc_conf,
        job_conf):
        """Performs only the forward pass for inference."""

        start_time = time.time()
        print("Testing started", flush=True, file=job_conf["log"][0])

        # set layers to inference mode if behavior differs between train and prediction
        # (prepares Dropout and BatchNormalization layers to enable and to freeze parameters, respectively)
        self.model.eval()

        # test the model on the test set
        top1_acc_val, top5_acc_val, ce_epoch_loss_val, mse_epoch_loss_val, duration_val = self._test(
            dataloader=dataloader,
            foo=self._forward,
            log=job_conf['log'])

        print(
            "[rank {0}]: completed testing"
            .format(self.rank),
            flush=True,
            file=job_conf['log'][0])

        loss_val = torch.tensor([ce_epoch_loss_val, mse_epoch_loss_val], device=self.rank)
        top1_acc_val = torch.tensor([top1_acc_val], device=self.rank)
        top5_acc_val = torch.tensor([top5_acc_val], device=self.rank)
        
        self._reduce_metrics()

        # save all metrics
        pd.DataFrame(data={"top1": top1_acc_val.cpu().numpy(), "top5": top5_acc_val.cpu().numpy()}, index=[0,1]).to_csv('{0}/accuracy.csv'.format(proc_conf['save_dir']))

        self._save_metrics(proc_conf['save_dir'], None)

        self._demo_segmentation_masks(
            dataloader=dataloader, 
            suffix=None,
            demo=proc_conf['demo'],
            save_dir=proc_conf['save_dir'])

        # log and send notifications
        print(
            "[test]: epoch_loss = {0}, top1_acc = {1}, top5_acc = {2}{3}"
            .format(
                (loss_val.sum() / len(dataloader)).cpu().numpy(),
                top1_acc_val.cpu().numpy(),
                top5_acc_val.cpu().numpy(),
                self._log_metrics()),
            flush=True,
            file=job_conf['log'][0])

        if job_conf['verbose'] > 0:
            print(
                "[test]: time = {1}"
                .format(duration_val),
                flush=True,
                file=job_conf['log'][0])

        # format a stats table (in newest to oldest order) and send it as email update
        if job_conf['verbose'] > 1:
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
                    job_conf['jobname'],
                    job_conf['email']))

        print("Testing completed in: {0}".format(time.time() - start_time), flush=True, file=job_conf["log"][0])

        return None


    def benchmark(
        self,
        dataloader,
        proc_conf,
        arch_conf,
        job_conf):
        """Benchmarks FP32 and INT8 (PTSQ) performance of a model.

        NOTE: currently only supports `original` and `realtime` on `dir` dataset types.
        """

        start_time = time.time()
        print("Benchmarking started", flush=True, file=job_conf["log"][0])

        # replace trainable layers of the proposed model with quantizeable inference-only version
        # TODO: extend to other models and call using meaningful common method name
        if proc_conf['model'] == 'realtime':
            self.model._swap_layers_for_inference()
            self.model.eval_()
        self.model.eval()
        
        # measure FP32 latency on CPU
        _, _, _, _, latency_fp32 = self._test(
            dataloader=dataloader,
            foo=self._forward_rt,
            log=job_conf['log'],
            num_samples=1)

        print(
            "[benchmark FP32]: {0} spf"
            .format(latency_fp32),
            flush=True,
            file=job_conf['log'][0])

        size_fp32 = self._model_size(proc_conf['save_dir'])

        # prepare the FP32 model: attaches observers to all layers, including the custom layer
        qconfig = torch.ao.quantization.get_default_qconfig_mapping(proc_conf['backend'])
        self.model = prepare_fx(self.model, qconfig_mapping=qconfig, example_inputs=torch.randn(1,3,1,25), prepare_custom_config=arch_conf['prepare_dict'])
        
        # calibrate the observed model (must be done on the same device as training)
        top1_acc_cal, top5_acc_cal, ce_epoch_loss_cal, mse_epoch_loss_cal, _ = self._test(
            dataloader=dataloader,
            foo=self._forward_rt,
            log=job_conf['log'],
            num_samples=1)

        self._save_metrics(proc_conf['save_dir'], "_fp32")

        self._demo_segmentation_masks(
            dataloader=dataloader,
            suffix="_fp32",
            demo=proc_conf["demo"],
            save_dir=proc_conf['save_dir'])

        # move the model to the CPU for INT8 inference
        self.model.to("cpu")

        # quantize the calibrated model
        self.model = convert_fx(self.model, qconfig_mapping=qconfig, convert_custom_config=arch_conf['convert_dict'])
        self.model.eval()

        self.model = torch.jit.script(self.model)

        # evaluate performance of the PTSQ model (must be done on the same device as training)
        # measure INT8 latency
        top1_acc_q, top5_acc_q, ce_epoch_loss_q, mse_epoch_loss_q, latency_int8 = self._test(
            dataloader=dataloader,
            foo=self._forward_rt,
            log=job_conf['log'],
            num_samples=1)

        print(
            "[benchmark INT8]: {0} spf"
            .format(latency_int8),
            flush=True,
            file=job_conf['log'][0])

        size_int8 = self._model_size(proc_conf['save_dir'])

        self._save_metrics(proc_conf['save_dir'], "_int8")

        self._demo_segmentation_masks(
            dataloader=dataloader, 
            suffix="_int8",
            demo=proc_conf["demo"],
            save_dir=proc_conf['save_dir'])

        # save all the measurements
        pd.DataFrame(
            data={
                'top1_fp32': top1_acc_cal,
                'top1_int8': top1_acc_q,
                'top5_fp32': top5_acc_cal,
                'top5_int8': top5_acc_q
            },
            index=[0]).to_csv('{0}/accuracy.csv'.format(proc_conf['save_dir']))

        pd.DataFrame(
            data={
                'ce_fp32': ce_epoch_loss_cal,
                'mse_fp32': mse_epoch_loss_cal,
                'ce_int8': ce_epoch_loss_q,
                'mse_int8': mse_epoch_loss_q,
            },
            index=[0]).to_csv('{0}/loss.csv'.format(proc_conf['save_dir']))

        pd.DataFrame(data={"latency_fp32": latency_fp32, "latency_int8": latency_int8},index=[0]).to_csv('{0}/latency.csv'.format(proc_conf['save_dir']))
        pd.DataFrame(data={"size_fp32": size_fp32, "size_int8": size_int8},index=[0]).to_csv('{0}/model-size.csv'.format(proc_conf['save_dir']))

        print("Benchmarking completed in: {0}".format(time.time() - start_time), flush=True, file=job_conf["log"][0])

        return None
