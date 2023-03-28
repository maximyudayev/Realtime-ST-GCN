import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import pandas as pd
import time
import os
from torch.autograd import Variable
import cProfile, pstats
from pstats import SortKey

class Processor:
    """ST-GCN processing wrapper for training and testing the model.

    Methods:
        train()
            Trains the model, given user-defined training parameters.

        test()
            Performs only the forward pass for inference.

    TODO:
        ``1.`` Provide useful prediction statistics (e.g. IoU, jitter, etc.).
    """

    def __init__(
        self,
        model,
        num_classes,
        dataloader,
        device):
        """
        Args:
            model : ``torch.nn.Module``
                Configured PyTorch model.
            
            num_classes : ``int``
                Number of action classification classes.

            dataloader : ``torch.utils.data.DataLoader``
                Data handle to account for its class imbalance in the CE loss.
        """

        classes = torch.tensor(range(num_classes), dtype=torch.float32)
        class_dist = torch.zeros(num_classes, dtype=torch.float32)

        for _, labels in dataloader:
            class_dist += torch.sum(
                (labels[:,:,None].to(torch.float32) == classes[None].expand(labels.shape[1],-1)).to(torch.float32),
                dim=(0,1))

        self.model = model
        self.ce = nn.CrossEntropyLoss(weight=(1-class_dist/torch.sum(class_dist)).to(device=device), reduction='mean')
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = num_classes


    def update_lr_(self, learning_rate, learning_rate_decay, epoch):
        """Decays learning rate monotonically by the provided factor."""
        
        rate = learning_rate * pow(learning_rate_decay, epoch)
        for g in self.optimizer.param_groups:
            g['lr'] = rate


    def forward_(
        self,
        captures,
        labels,
        device,
        **kwargs):
        """Does the forward pass on the model.
        
        If `dataset_type` is `'dir'`, processes 1 trial at a time, chops each sequence 
        into equal segments that are split across available executors (GPUs) for parallel computation.
        
        If `model` is `'original'` and `latency` is `True`, applies the original classification model
        on non-overlapping windows of size `receptive_field` over the input stream, producing outputs at a 
        reduced temporal resolution inversely proportional to the size of the window. Trades prediction
        resolution for compute (does not compute redundant values for input frames otherwise overlapped by 
        multiple windows).
        """

        # move both data to the compute device
        # (captures is a batch of full-length captures, label is a batch of ground truths)
        captures, labels = captures.to(device), labels.to(device)

        N, _, L, _ = captures.size()

        # Splits trial into overlapping subsequences of samples
        if kwargs['dataset_type'] == 'dir': 
            if kwargs['model'] == 'original':
                # zero pad the input across time from start by the receptive field size
                captures = F.pad(captures, (0, 0, kwargs['receptive_field']-1, 0))
                stride = kwargs['receptive_field'] if kwargs['latency'] else 1
                captures = captures.unfold(2, kwargs['receptive_field'], stride)
                labels = labels[:, ::stride]
            else:
                # Size to divide the trial into to construct a data parallel batch
                # NOTE: adjust if kernel is different in multi-stage ST-GCN
                W = math.ceil((L-(kwargs['kernel'][0]-1))/kwargs['segment'])
                # if captures is perfectly unfolded without padding, P will be 0
                P = W*kwargs['segment']+(kwargs['kernel'][0]-1)-L
                # Pad the end of the sequence to use all of the available readings (masks actual outputs later)
                captures = F.pad(captures, (0, 0, 0, P))
                captures = captures.unfold(2, W+(kwargs['kernel'][0]-1), W)
            
            N, C, N_new, V, T_new = captures.size()
            # (N,C,N',V,T') -> batches of unfolded slices
            # .contiguous() is needed before .view(), but also after .unfold() to operate on same data element 
            # in the overlapping segments. Otherwise two segments will update the same memory location, leaking data.
            # No need to .clone() the sliced view of the tensor after .contiguous()
            captures = captures.permute(0, 2, 1, 4, 3).contiguous()
            captures = captures.view(N * N_new, C, T_new, V)
            # (N'',C,T',V)

        # make predictions and compute the loss
        # forward pass the minibatch through the model for the corresponding subject
        # the input tensor has shape (N, V, C, L): N-batch, V-nodes, C-channels, L-length
        # the output tensor has shape (N, C', L)
        predictions = self.model(Variable(captures, requires_grad=True))

        if kwargs['dataset_type'] == 'dir':
            C_new = predictions.size(1)
            if kwargs['model'] == 'original':
                # arrange tensor back into a time series
                predictions = predictions.view(N, N_new, C_new)
                predictions = predictions.permute(0, 2, 1)
            else:
                # clear the overlapping Gamma-1 predictions at the start of each segment (except the very first segment), since 
                # overlapped regions are added when folding the tensor
                predictions[1:,:,:kwargs['kernel'][0]-1] = 0
                # shuffle data around for the correct contiguous access by the fold()
                predictions = predictions[None].permute(0, 2, 3, 1).contiguous()
                predictions = predictions.view(N, C_new * (W+(kwargs['kernel'][0]-1)), -1)
                # fold segments of the original trial computed in parallel on multiple executors back into original length sequence
                # and drop the end padding used to fill tensor to equal row-column size
                predictions = F.fold(
                    predictions,
                    output_size=(1, L+P),
                    kernel_size=(1, W+(kwargs['kernel'][0]-1)),
                    stride=(1, W))[:,:,0,:L]

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
        top5_cor = torch.sum(top5_predicted == labels[:,None,:]).data.item()
        tot = labels.numel()

        return top1_predicted, top5_predicted, top1_cor, top5_cor, tot, ce, mse


    def validate_(
        self,
        dataloader,
        device,
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

            confusion_matrix = torch.zeros(self.num_classes, self.num_classes, device=device, dtype=torch.int64)
            
            ce_epoch_loss_val = 0
            mse_epoch_loss_val = 0

            # sweep through the training dataset in minibatches
            for captures, labels in dataloader:
                top1_predicted, _, top1_cor, top5_cor, tot, ce, mse = self.forward_(captures, labels, device, **kwargs)

                top1_correct += top1_cor
                top5_correct += top5_cor
                total += tot

                # to calculate loss correctly, account for non-overlapping original model with reduced temporal resolution
                if kwargs['model'] == 'original':
                    labels = labels[:, ::kwargs['receptive_field'] if kwargs['latency'] else 1]
                
                N, L = labels.size()
                
                # epoch loss has to multiply by minibatch size to get total non-averaged loss, 
                # which will then be averaged across the entire dataset size, since
                # loss for dataset with equal-length trials averages the CE and MSE losses for each minibatch
                # (used for statistics)
                ce_epoch_loss_val += (ce*N).data.item()
                mse_epoch_loss_val += (mse*N).data.item()

                # collect the correct predictions for each class and total per that class
                # for batch_el in range(N*M):
                for batch_el in range(N):
                    # OHE 3D matrix, where label and prediction at time `t` are indices
                    top1_ohe = torch.zeros(L, self.num_classes, self.num_classes, device=device, dtype=torch.bool)
                    top1_ohe[range(L), top1_predicted[batch_el], labels[batch_el]] = True

                    # sum-reduce OHE 3D matrix to get number of true vs. false classifications for each class on this sample
                    confusion_matrix += torch.sum(top1_ohe, dim=0)

            test_end_time = time.time()

            top1_acc = top1_correct / total
            top5_acc = top5_correct / total
            duration = test_end_time - test_start_time

        return top1_acc, top5_acc, duration, confusion_matrix, ce_epoch_loss_val, mse_epoch_loss_val


    def train(
        self, 
        save_dir, 
        train_dataloader,
        val_dataloader,
        device,
        epochs,
        checkpoints,
        checkpoint,
        learning_rate,
        learning_rate_decay,
        **kwargs):
        """Trains the model, given user-defined training parameters."""

        # move the model to the compute device(s) if available (CPU, GPU, TPU, etc.)
        if torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "allocated GPUs", flush=True, file=kwargs['log'][0])
            self.model = nn.DataParallel(self.model)
        self.model.to(device)

        # setup the optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        if checkpoint:
            state = torch.load(checkpoint, map_location=device)
            range_epochs = range(state['epoch']+1, epochs)
            
            # load the checkpoint if not training from scratch
            self.optimizer.load_state_dict(state['optimizer_state_dict'])
        else:
            range_epochs = range(epochs)

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

        # train the model for num_epochs
        # (dataloader is automatically shuffled after each epoch)
        for epoch in range_epochs:
            # set layers to training mode if behavior of any differs between train and prediction
            # (prepares Dropout and BatchNormalization layers to disable and to learn parameters, respectively)
            self.model.train()

            confusion_matrix_train = torch.zeros(self.num_classes, self.num_classes, device=device, dtype=torch.int64)

            ce_epoch_loss_train = 0
            mse_epoch_loss_train = 0

            top1_correct = 0
            top5_correct = 0
            total = 0

            # decay learning rate every 10 epochs [ref: Yan 2018]
            if (epoch % 10 == 0):
                self.update_lr_(learning_rate, learning_rate_decay, epoch//10)

            epoch_start_time = time.time()

            self.optimizer.zero_grad()
            
            # sweep through the training dataset in minibatches
            for i, (captures, labels) in enumerate(train_dataloader):
                N, _, L, _ = captures.size()

                top1_predicted, _, top1_cor, top5_cor, tot, ce, mse = self.forward_(captures, labels, device, **kwargs)

                top1_correct += top1_cor
                top5_correct += top5_cor
                total += tot

                # epoch loss has to multiply by minibatch size to get total non-averaged loss, 
                # which will then be averaged across the entire dataset size, since
                # loss for dataset with equal-length trials averages the CE and MSE losses for each minibatch
                # (used for statistics)
                ce_epoch_loss_train += (ce*N).data.item()
                mse_epoch_loss_train += (mse*N).data.item()

                # loss is already a mean across minibatch for tensor of equally long trials, but
                # not for different-length trials -> needs averaging
                loss = ce + mse
                if (kwargs['dataset_type'] == 'dir' and
                    (len(train_dataloader) % kwargs['batch_size'] == 0 or
                    i < len(train_dataloader) - (len(train_dataloader) % kwargs['batch_size']))):
                    
                    # if the minibatch is the same size as requested (first till one before last minibatch)
                    # (because dataset is a multiple of batch size or if current minibatch is of requested size)
                    loss /= kwargs['batch_size']
                elif (kwargs['dataset_type'] == 'dir'and 
                    (len(train_dataloader) % kwargs['batch_size'] != 0 and
                    i >= len(train_dataloader) - (len(train_dataloader) % kwargs['batch_size']))):

                    # if the minibatch is smaller than requested (last minibatch)
                    loss /= (len(train_dataloader) % kwargs['batch_size'])

                # backward pass to compute the gradients
                loss.backward()

                # zero the gradient buffers after every batch
                # if dataset is a tensor with equal length trials, always enters
                # if dataset is a set of different length trials, enters every ``batch_size`` iteration
                if ((kwargs['dataset_type'] == 'dir' and
                        ((i + 1) % kwargs['batch_size'] == 0 or 
                        (i + 1) == len(train_dataloader))) or
                    (kwargs['dataset_type'] == 'file')):

                    # update parameters based on the computed gradients
                    self.optimizer.step()

                    # clear the gradients
                    self.optimizer.zero_grad()

                # to calculate confusion matrix correctly, account for non-overlapping original model with reduced temporal resolution
                if kwargs['model'] == 'original':
                    labels = labels[:, ::kwargs['receptive_field'] if kwargs['latency'] else 1]
                
                N, L = labels.size()

                # collect the correct predictions for each class and total per that class
                # for batch_el in range(N*M):
                for batch_el in range(N):
                    # OHE 3D matrix, where label and prediction at time `t` are indices
                    top1_ohe = torch.zeros(L, self.num_classes, self.num_classes, device=device, dtype=torch.bool)
                    top1_ohe[range(L), top1_predicted[batch_el], labels[batch_el]] = True

                    # sum-reduce OHE 3D matrix to get number of true vs. false classifications for each class on this sample
                    confusion_matrix_train += torch.sum(top1_ohe, dim=0)

            epoch_end_time = time.time()
            duration_train = epoch_end_time - epoch_start_time
            top1_acc_train = top1_correct / total
            top5_acc_train = top5_correct / total
            
            # save prediction and ground truth of reference sample
            ref_capture, ref_label = train_dataloader.dataset.__getitem__(328) # 328
            ref_predicted, _, _, _, _, _, _ = self.forward_(ref_capture[None], ref_label[None], device, **kwargs)
            pd.DataFrame(torch.stack((ref_label, ref_predicted[0].cpu())).numpy()).to_csv('{0}/segmentation_epoch-{1}_train.csv'.format(save_dir, epoch))

            # checkpoint the model during training at specified epochs
            if epoch in checkpoints:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "loss": (ce_epoch_loss_train + mse_epoch_loss_train) / len(train_dataloader),
                    }, "{0}/epoch-{1}.pt".format(save_dir, epoch))
            
            # set layers to inference mode if behavior differs between train and prediction
            # (prepares Dropout and BatchNormalization layers to enable and to freeze parameters, respectively)
            self.model.eval()

            # test the model on the validation set
            # will complain on CUDA devices that input gradients are none: irrelevant because it is a side effect of
            # the shared `forward_()` routine for both tasks, where the model is set to `train()` or `eval()` in the
            # corresponding caller function
            top1_acc_val, top5_acc_val, duration_val, confusion_matrix, ce_epoch_loss_val, mse_epoch_loss_val = self.validate_(
                dataloader=val_dataloader,
                device=device,
                **kwargs)

            # record all stats of interest for logging/notification
            epoch_list.insert(0, epoch)

            ce_loss_train_list.insert(0, ce_epoch_loss_train / len(train_dataloader))
            mse_loss_train_list.insert(0, mse_epoch_loss_train / len(train_dataloader))
            epoch_loss_train_list.insert(0, (ce_epoch_loss_train + mse_epoch_loss_train) / len(train_dataloader))

            ce_loss_val_list.insert(0, ce_epoch_loss_val / len(val_dataloader))
            mse_loss_val_list.insert(0, mse_epoch_loss_val / len(val_dataloader))
            epoch_loss_val_list.insert(0, (ce_epoch_loss_val + mse_epoch_loss_val) / len(val_dataloader))

            top1_acc_train_list.insert(0, top1_acc_train)
            top1_acc_val_list.insert(0, top1_acc_val)
            top5_acc_train_list.insert(0, top5_acc_train)
            top5_acc_val_list.insert(0, top5_acc_val)            
            duration_train_list.insert(0, duration_train)
            duration_val_list.insert(0, duration_val)

            # save prediction and ground truth of reference sample
            ref_capture, ref_label = train_dataloader.dataset.__getitem__(328)
            ref_predicted, _, _, _, _, _, _ = self.forward_(ref_capture[None], ref_label[None], device, **kwargs)
            pd.DataFrame(torch.stack((ref_label, ref_predicted[0].cpu())).numpy()).to_csv('{0}/segmentation_epoch-{1}_val.csv'.format(save_dir, epoch))

            # save confusion matrix as a CSV file
            pd.DataFrame(confusion_matrix.cpu().numpy()).to_csv('{0}/confusion_matrix_epoch-{1}.csv'.format(save_dir, epoch))
            pd.DataFrame(confusion_matrix_train.cpu().numpy()).to_csv('{0}/confusion_matrix_epoch-{1}_train.csv'.format(save_dir, epoch))

            # log and send notifications
            print(
                "[epoch {0}]: epoch_loss = {1}, top1_acc_train = {2}, top5_acc_train = {3}, top1_acc_val = {4}, top5_acc_val = {5}"
                .format(
                    epoch, 
                    (ce_epoch_loss_train + mse_epoch_loss_train) / len(train_dataloader),
                    top1_acc_train,
                    top5_acc_train,
                    top1_acc_val,
                    top5_acc_val),
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
                    'header="\n %-6s %5s %11s %11s %9s %9s %11s %9s\n";'
                    'format=" %-03d %4.6f %1.4f %1.4f %1.4f %1.4f %5.6f %5.6f\n";'
                    'printf "$header" "EPOCH" "LOSS" "TOP1_TRAIN" "TOP5_TRAIN" "TOP1_VAL" "TOP5_VAL" "TIME_TRAIN" "TIME_VAL" > $PBS_O_WORKDIR/mail_draft_{1}.txt;'
                    'printf "$format" {0} >> $PBS_O_WORKDIR/mail_draft_{1}.txt;'
                    'cat $PBS_O_WORKDIR/mail_draft_{1}.txt | mail -s "[{1}]: status update" {2}'
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
                        '_'.join([kwargs['model'], 'red' if kwargs['model'] == 'original' and kwargs['latency'] else '', *kwargs['jobname']]),
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
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": (ce_epoch_loss_train + mse_epoch_loss_train) / len(train_dataloader),
            }, "{0}/final.pt".format(save_dir))

        return


    def test(
        self,
        save_dir,
        dataloader,
        device,
        **kwargs):
        """Performs only the forward pass for inference.
        """
        
        # set layers to inference mode if behavior differs between train and prediction
        # (prepares Dropout and BatchNormalization layers to enable and to freeze parameters, respectively)
        self.model.eval()
        
        # move the model to the compute device(s) if available (CPU, GPU, TPU, etc.)
        if torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "allocated GPUs", flush=True, file=kwargs['log'][0])
            self.model = nn.DataParallel(self.model)
        self.model.to(device)

        # test the model on the validation set
        top1_acc_val, top5_acc_val, duration_val, confusion_matrix, ce_epoch_loss_val, mse_epoch_loss_val = self.validate_(
            dataloader=dataloader,
            device=device,
            **kwargs)
        
        # save confusion matrix as a CSV file
        pd.DataFrame(confusion_matrix.cpu().numpy()).to_csv('{0}/confusion_matrix.csv'.format(save_dir))

        # log and send notifications
        print(
            "[test]: epoch_loss = {0}, top1_acc = {1}, top5_acc = {2}"
            .format(
                (ce_epoch_loss_val + mse_epoch_loss_val) / len(dataloader),
                top1_acc_val,
                top5_acc_val),
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


    def benchmark(
        self,
        save_dir,
        dataloader,
        device,
        **kwargs):
        """Benchmarks realtime inference for a model.

        NOTE: currently only supports `original` and `realtime` on `dir` dataset types.
        """

        # set layers to inference mode if behavior differs between train and prediction
        # (prepares Dropout and BatchNormalization layers to enable and to freeze parameters, respectively)
        self.model.eval()

        # move the model to the compute device(s) if available (CPU, GPU, TPU, etc.)
        if torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "allocated GPUs", flush=True, file=kwargs['log'][0])
            self.model = nn.DataParallel(self.model)
        self.model.to(device)

        # do not record gradients
        with torch.no_grad():
            top1_correct = 0
            top5_correct = 0
            total = 0

            # confusion matrix for frame-wise metrics
            confusion_matrix = torch.zeros(self.num_classes, self.num_classes, device=device, dtype=torch.int64)
            
            # stats for F1@k segmentation-detection metric of Lea, et al. (2016)
            overlap = torch.tensor([0.1, 0.25, 0.5], device=device, dtype=torch.float32)
            f1 = torch.zeros(len(dataloader), overlap.size(0), device=device, dtype=torch.float32)
            
            # stats for segmental edit score
            edit_score = 0

            # sweep through the training dataset in minibatches
            # calculate IoU for the F1@k metrics
            # NOTE: (assumes N=1)
            for j, (captures, labels) in enumerate(dataloader):
                top1_predicted, _, top1_cor, top5_cor, tot, _, _ = self.forward_(captures, labels, device, **kwargs)

                top1_correct += top1_cor
                top5_correct += top5_cor
                total += tot

                labels = labels.to(device)
                N, L = labels.size()

                # detects edges of ground truth
                # (will yield non-zero values at the edges of classes)
                edges_labels = torch.zeros(L, device=device, dtype=torch.int64)
                edges_labels[0] = 1
                edges_labels[1:] = labels[0,1:]-labels[0,:-1]
                
                edges_indices_labels = edges_labels.nonzero()[:,0]
                edges_indices_labels_shifted = torch.zeros_like(edges_indices_labels, device=device, dtype=torch.int64)
                edges_indices_labels_shifted[:-1] = edges_indices_labels[1:]
                edges_indices_labels_shifted[-1] = L

                label_segments_used = torch.zeros(edges_indices_labels.size(0), overlap.size(0), device=device, dtype=torch.bool)

                # detects edges of predictions
                edges_predictions = torch.zeros(L, device=device, dtype=torch.int64)
                edges_predictions[0] = 1
                edges_predictions[1:] = top1_predicted[0,1:]-top1_predicted[0,:-1]

                edges_indices_predictions = edges_predictions.nonzero()[:,0]
                edges_indices_predictions_shifted = torch.zeros_like(edges_indices_predictions, device=device, dtype=torch.int64)
                edges_indices_predictions_shifted[:-1] = edges_indices_predictions[1:]
                edges_indices_predictions_shifted[-1] = L

                tp = torch.zeros(self.num_classes, overlap.size(0), device=device, dtype=torch.int64)
                fp = torch.zeros(self.num_classes, overlap.size(0), device=device, dtype=torch.int64)

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
                    IoU = (intersection/union)*(top1_predicted[0,edges_indices_predictions[i]]==labels[0,edges_indices_labels])
                    # ground truth segment with the largest IoU is the (potential) hit
                    idx = IoU.argmax()

                    # predicted segment is a hit if it exceeds IoU threshold and if its label has not been matched against yet
                    hits = torch.bitwise_and(IoU[idx].gt(overlap),torch.bitwise_not(label_segments_used[idx]))

                    # mark TP and FP correspondingly
                    # correctly classified, exceeding the threshold and the first predicted segment to match the ground truth
                    tp[top1_predicted[0,edges_indices_predictions[i]]] += hits
                    # correctly classified, but under the threshold or not the first predicted segment to match the ground truth
                    fp[top1_predicted[0,edges_indices_predictions[i]]] += torch.bitwise_not(hits)
                    # mark ground truth segment used if marked TP
                    label_segments_used[idx] += hits

                TP = tp.sum(dim=0)
                FP = fp.sum(dim=0)
                # FN are unmatched ground truth segments (misses)
                FN = label_segments_used.size(0) - label_segments_used.sum(dim=0)

                # calculate the F1 score
                precision = TP / (TP+FP)
                recall = TP / (TP+FN)
                f1[j] = 2*precision*recall/(precision+recall)

                # collect the segmental edit score
                m_row = edges_indices_predictions.size(0)
                n_col = edges_indices_labels.size(0)

                D = torch.zeros(m_row+1, n_col+1, device=device, dtype=torch.float32)
                D[:,0] = torch.arange(m_row+1)
                D[0,:] = torch.arange(n_col+1)

                for j in range(1, n_col+1):
                    for i in range(1, m_row+1):
                        if labels[0,edges_indices_labels][j-1] == top1_predicted[0,edges_indices_predictions][i-1]:
                            D[i, j] = D[i - 1, j - 1]
                        else:
                            D[i, j] = min(D[i - 1, j] + 1,
                                        D[i, j - 1] + 1,
                                        D[i - 1, j - 1] + 1)

                edit_score += (1 - D[-1, -1] / max(m_row, n_col))

                # collect the correct predictions for each class and total per that class
                # for batch_el in range(N*M):
                for batch_el in range(N):
                    # OHE 3D matrix, where label and prediction at time `t` are indices
                    top1_ohe = torch.zeros(L, self.num_classes, self.num_classes, device=device, dtype=torch.bool)
                    top1_ohe[range(L), top1_predicted[batch_el], labels[batch_el]] = True

                    # sum-reduce OHE 3D matrix to get number of true vs. false classifications for each class on this sample
                    confusion_matrix += torch.sum(top1_ohe, dim=0)

            top1_acc = top1_correct / total
            top5_acc = top5_correct / total
            # discard NaN F1 values and compute the macro F1-score (average)
            F1 = f1.nan_to_num(0).mean(dim=0)
            edit_score /= len(dataloader)

            pd.DataFrame(torch.stack((overlap,F1)).cpu().numpy()).to_csv('{0}/macro-F1@k.csv'.format(save_dir))
            pd.DataFrame(data={"top1": top1_acc, "top5": top5_acc}, index=[0]).to_csv('{0}/accuracy.csv'.format(save_dir))
            pd.DataFrame(data={"edit": edit_score.cpu().numpy()}, index=[0]).to_csv('{0}/edit.csv'.format(save_dir))
            # save confusion matrix as a CSV file
            pd.DataFrame(confusion_matrix.cpu().numpy()).to_csv('{0}/confusion_matrix.csv'.format(save_dir))

            # sweep through the sample trials
            for i in [175, 293, 37]: # 39 (L=4718), 177 (L=6973), 299 (L=2378)
                # save prediction and ground truth of reference samples
                captures, labels = dataloader.dataset.__getitem__(i)

                # move both data to the compute device
                # (captures is a batch of full-length captures, label is a batch of ground truths)
                captures, labels = captures[None].to(device), labels[None].to(device)

                N, _, L, _ = captures.size()

                top1_predicted, _, _, _, _, _, _ = self.forward_(captures, labels, device, **kwargs)

                pd.DataFrame(torch.stack((labels[0], top1_predicted[0])).cpu().numpy()).to_csv('{0}/segmentation-{1}.csv'.format(save_dir, i))

            # # timing benchmarking
            # latency = 0
            # if kwargs['model'] == 'realtime':
            #     self.model._swap_layers_for_inference()

            # # Splits trial for `original` model into overlapping subsequences of samples to separately feed into the model
            # if kwargs['model'] == 'original':
            #     # zero pad the input across time from start by the receptive field size
            #     captures = F.pad(captures, (0, 0, kwargs['receptive_field']-1, 0))
            #     stride = kwargs['receptive_field'] if kwargs['latency'] else 1
            #     captures = captures.unfold(2, kwargs['receptive_field'], stride)
            #     labels = labels[:, ::stride]
            
            #     N, C, N_new, V, T_new = captures.size()
            #     # (N,C,N',V,T') -> batches of unfolded slices
            #     # .contiguous() is needed before .view(), but also after .unfold() to operate on same data element 
            #     # in the overlapping segments. Otherwise two segments will update the same memory location, leaking data.
            #     # No need to .clone() the sliced view of the tensor after .contiguous()
            #     captures = captures.permute(0, 2, 1, 4, 3).contiguous()
            #     captures = captures.view(N * N_new, C, T_new, V)
            #     # (N'',C,T',V)

            #     with cProfile.Profile() as pr:
            #         for j in range(N_new):
            #             start = time.time()

            #             # the input tensor has shape (N, V, C, L): N-batch, V-nodes, C-channels, L-length
            #             # the output tensor has shape (N, C', L)
            #             self.model(captures[j:j+1])
                        
            #             lat += (time.time() - start)

            #         pstats.Stats(pr).sort_stats(SortKey.TIME).print_stats()

            #     latency += lat / N_new
            # else:
            #     with cProfile.Profile() as pr:
            #         for j in range(L):
            #             start = time.time()

            #             # the input tensor has shape (N, V, C, L): N-batch, V-nodes, C-channels, L-length
            #             # the output tensor has shape (N, C', L)
            #             self.model(captures[:,:,j:j+1,:])
                        
            #             lat += (time.time() - start)
                
            #         pstats.Stats(pr).sort_stats(SortKey.TIME).print_stats()

            #     latency += lat / L

            # print(
            #     "[benchmark]: {0} spf"
            #     .format(latency/(i+1)),
            #     flush=True,
            #     file=kwargs['log'][0])

        return
