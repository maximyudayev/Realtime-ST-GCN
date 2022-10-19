import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import pandas as pd
import time
import os
    

def validate_(
    model,
    num_classes,
    dataloader,
    device,
    **kwargs):
    """Routine for model validation.

    Shared between train and test scripts: train invokes it after each epoch trained,
    test invokes it once for inference only.
    """

    # # sets all layers into evaluation mode except the dropout layers
    # for layer in model.modules():
    #     if isinstance(layer, nn.Dropout):
    #         layer.train()

    # do not record gradients
    with torch.no_grad():    
        top1_correct = 0
        top5_correct = 0
        total = 0

        test_start_time = time.time()

        confusion_matrix = torch.zeros(num_classes, num_classes, device=device)
        total_per_class = torch.zeros(num_classes, 1, device=device)

        # sweep through the training dataset in minibatches
        for captures, labels in dataloader:
            N, _, L, _ = captures.size()
            # move both data to the compute device
            # (captures is a batch of full-length captures, label is a batch of ground truths)
            captures, labels = captures.to(device), labels.to(device)

            # make predictions and compute the loss
            # forward pass the minibatch through the model for the corresponding subject
            # the input tensor has shape (N, C, L, V): N-batch, C-channels, L-length, V-nodes
            # the output tensor has shape (N, C, L)
            predictions = model(captures)
            
            # calculate the predictions statistics
            # this only sums the number of top-1 correctly predicted frames, but doesn't look at prediction jitter
            _, top5_predicted = torch.topk(predictions, k=5, dim=1)
            top1_predicted = top5_predicted[:,0,:]

            top1_cor = torch.sum(top1_predicted == labels).data.item()
            top1_correct += top1_cor
            top5_cor = torch.sum(top5_predicted == labels[:,None,:]).data.item()
            top5_correct += top5_cor

            tot = labels.numel()
            total += tot

            # collect the correct predictions for each class and total per that class
            # for batch_el in range(N*M):
            for batch_el in range(N):
                top1_predicted_ohe = torch.zeros(L, num_classes, device=device)
                top1_predicted_ohe[range(L), top1_predicted[batch_el]] = 1
                confusion_matrix[labels[batch_el, 0]] += top1_predicted_ohe.sum(dim=0)
                total_per_class[labels[batch_el, 0]] += L

        test_end_time = time.time()

        # normalize each row of the confusion matrix to obtain class probabilities
        confusion_matrix = torch.div(confusion_matrix, total_per_class)

        top1_acc = top1_cor / total
        top5_acc = top5_cor / total
        duration = test_end_time - test_start_time

    return top1_acc, top5_acc, duration, confusion_matrix


class Processor:
    """ST-GCN processing wrapper for training and testing the model.

    Methods:
        train()
            Trains the model, given user-defined training parameters.

        predict()
            Performs only the forward pass for inference.
    """

    def __init__(
        self,
        model,
        num_classes):
        """
        Args:
            model : ``torch.nn.Module``
                Configured PyTorch model.
            
            num_classes : ``int``
                Number of action classification classes.
        """

        self.model = model
        self.ce = nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')
        self.mse = nn.MSELoss(reduction='mean')
        self.num_classes = num_classes


    def update_lr(self, learning_rate, learning_rate_decay, epoch):
        """Decays learning rate monotonically by the provided factor."""
        
        rate = learning_rate * pow(learning_rate_decay, epoch)
        for g in self.optimizer.param_groups:
            g['lr'] = rate


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
        """Trains the model, given user-defined training parameters.

        TODO:
            ``1.`` Provide useful prediction statistics (e.g. IoU, jitter, etc.).
        """

        # move the model to the compute device(s) if available (CPU, GPU, TPU, etc.)
        if torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "allocated GPUs", flush=True, file=kwargs['log'][0])
            self.model = nn.DataParallel(self.model)
        self.model.to(device)

        if checkpoint:
            state = torch.load(checkpoint, map_location=device)
            range_epochs = range(state['epoch']+1, epochs)
        else:
            range_epochs = range(epochs)

        # setup the optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        # load the checkpoint if not training from scratch
        if checkpoint:
            self.optimizer.load_state_dict(state['optimizer_state_dict'])

        # variables for email updates
        epoch_list = []
        epoch_loss_list = []
        top1_acc_train_list = []
        top5_acc_train_list = []
        duration_train_list = []
        top1_acc_val_list = []
        top5_acc_val_list = []
        duration_val_list = []

        # train the model for num_epochs
        # (dataloader is automatically shuffled after each epoch)
        for epoch in range_epochs:
            # set layers to training mode if behavior of any differs between train and prediction
            # (prepares Dropout and BatchNormalization layers to disable and to learn parameters, respectively)
            self.model.train()

            epoch_loss = 0
            top1_correct = 0
            top5_correct = 0
            total = 0

            # decay learning rate every 10 epochs [ref: Yan 2018]
            if (epoch % 10 == 0):
                self.update_lr(learning_rate, learning_rate_decay, epoch//10)

            epoch_start_time = time.time()

            # sweep through the training dataset in minibatches
            for captures, labels in train_dataloader:
                N, _, L, _ = captures.size()
                # move both data to the compute device
                # (captures is a batch of full-length captures, label is a batch of ground truths)
                captures, labels = captures.to(device), labels.to(device)
                
                # zero the gradient buffers
                self.optimizer.zero_grad()
                
                # make predictions and compute the loss
                # forward pass the minibatch through the model for the corresponding subject
                # the input tensor has shape (N, C, L, V): N-batch, C-channels, L-length, V-nodes
                # the output tensor has shape (N, C, L)
                predictions = self.model(captures)

                # cross-entropy expects output as class indices (N, C, K), with labels (N, K): 
                # N-batch (flattened multi-skeleton minibatch), C-class, K-extra dimension (capture length)
                # CE + MSE loss metric tuning is taken from @BenjaminFiltjens's MS-GCN:
                # CE guides model toward absolute correctness on single frame predictions,
                # MSE component punishes large variations in class probabilities between consecutive samples
                loss = self.ce(predictions, labels)
                loss += 0.15 * torch.mean(
                    torch.clamp(
                        self.mse(
                            F.log_softmax(predictions[:,:,1:], dim=1), 
                            F.log_softmax(predictions.detach()[:,:,:-1], dim=1)),
                        min=0,
                        max=16))

                epoch_loss += loss.data.item()
                
                # backward pass to compute the gradients
                loss.backward()

                # update parameters based on the computed gradients
                self.optimizer.step()

                # calculate the predictions statistics
                # this only sums the number of top-1/5 correctly predicted frames, but doesn't look at prediction jitter
                _, top5_predicted = torch.topk(predictions, k=5, dim=1)
                top1_predicted = top5_predicted[:,0,:]

                top1_cor = torch.sum(top1_predicted == labels).data.item()
                top1_correct += top1_cor
                top5_cor = torch.sum(top5_predicted == labels[:,None,:]).data.item()
                top5_correct += top5_cor

                tot = labels.numel()
                total += tot

            epoch_end_time = time.time()
            duration_train = epoch_end_time - epoch_start_time
            top1_acc_train = top1_correct / total
            top5_acc_train = top5_correct / total
            
            # checkpoint the model during training at specified epochs
            if epoch in checkpoints:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "loss": epoch_loss,
                    }, "{0}/epoch-{1}.pt".format(save_dir, epoch))
            
            # set layers to inference mode if behavior differs between train and prediction
            # (prepares Dropout and BatchNormalization layers to enable and to freeze parameters, respectively)
            self.model.eval()

            # test the model on the validation set
            top1_acc_val, top5_acc_val, duration_val, confusion_matrix = validate_(
                model=self.model, 
                num_classes=self.num_classes,
                dataloader=val_dataloader,
                device=device)

            # save confusion matrix as a CSV file
            pd.DataFrame(confusion_matrix.cpu().numpy()).to_csv('{0}/confusion_matrix_epoch-{1}.csv'.format(save_dir, epoch))

            # log and send notifications
            print(
                "[epoch {0}]: epoch loss = {1}, top1_acc_train = {2}, top5_acc_train = {3}, top1_acc_val = {4}, top5_acc_val = {5}"
                .format(
                    epoch, 
                    epoch_loss / len(train_dataloader),
                    top1_acc_train,
                    top5_acc_train,
                    top1_acc_val,
                    top5_acc_val),
                flush=True,
                file=kwargs['log'][0])
            print(
                "[epoch {0}]: train_time = {1}, val_time = {2}"
                .format(
                    epoch,
                    duration_train,
                    duration_val),
                flush=True,
                file=kwargs['log'][0])

            # send an email update
            epoch_list.insert(0, epoch)
            epoch_loss_list.insert(0, epoch_loss / len(train_dataloader))
            top1_acc_train_list.insert(0, top1_acc_train)
            top5_acc_train_list.insert(0, top5_acc_train)
            duration_train_list.insert(0, duration_train)
            top1_acc_val_list.insert(0, top1_acc_val)
            top5_acc_val_list.insert(0, top5_acc_val)
            duration_val_list.insert(0, duration_val)

            # format a stats table (in newest to oldest order) and send it by email
            os.system(
                'header="\n %-6s %5s %11s %11s %9s %9s %11s %9s\n";'
                'format=" %-03d %4.6f %1.4f %1.4f %1.4f %1.4f %5.6f %5.6f\n";'
                'printf "$header" "EPOCH" "LOSS" "TOP1_TRAIN" "TOP5_TRAIN" "TOP1_VAL" "TOP5_VAL" "TIME_TRAIN" "TIME_VAL" > $PBS_O_WORKDIR/mail_draft_{1}.txt;'
                'printf "$format" {0} >> $PBS_O_WORKDIR/mail_draft_{1}.txt;'
                'cat $PBS_O_WORKDIR/mail_draft_{1}.txt | mail -s "[{1}]: $PBS_JOBNAME status update" {2}'
                .format(
                    ' '.join([
                        ' '.join([str(e) for e in t]) for t 
                        in zip(
                            epoch_list,
                            epoch_loss_list,
                            top1_acc_train_list,
                            top5_acc_train_list,
                            top1_acc_val_list,
                            top5_acc_val_list,
                            duration_train_list,
                            duration_val_list)]),
                    os.getenv('PBS_JOBID').split('.')[0],
                    kwargs['email']))

            # save (update) train-validation curve as a CSV file after each epoch
            pd.DataFrame(
                data={
                    'top1_train': top1_acc_train_list,
                    'top1_val': top1_acc_val_list,
                    'top5_train': top5_acc_train_list,
                    'top5_val': top5_acc_val_list
                }).to_csv('{0}/train-validation-curve.csv'.format(save_dir))

        # save the final model
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": epoch_loss,
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
        top1_acc_val, top5_acc_val, duration_val, confusion_matrix = validate_(
            model=self.model, 
            num_classes=self.num_classes,
            dataloader=dataloader,
            device=device)
        
        # save confusion matrix as a CSV file
        pd.DataFrame(confusion_matrix.cpu().numpy()).to_csv('{0}/confusion_matrix.csv'.format(save_dir))

        # log and send notifications
        print(
            "[test]: top1_acc = {0}, top5_acc = {1}"
            .format( 
                top1_acc_val,
                top5_acc_val),
            flush=True,
            file=kwargs['log'][0])
        print(
            "[test]: time = {1}"
            .format(duration_val),
            flush=True,
            file=kwargs['log'][0])

        # format a stats table (in newest to oldest order) and send it by email
        os.system(
            'header="\n %-5s %5s %5s\n";'
            'format=" %-1.4f %1.4f %5.6f\n";'
            'printf "$header" "TOP1" "TOP5" "TIME" > $PBS_O_WORKDIR/mail_draft_{1}.txt;'
            'printf "$format" {0} >> $PBS_O_WORKDIR/mail_draft_{1}.txt;'
            'cat $PBS_O_WORKDIR/mail_draft_{1}.txt | mail -s "[{1}]: $PBS_JOBNAME status update" {2}'
            .format(
                ' '.join([
                    ' '.join([str(e) for e in t]) for t 
                    in zip(
                        top1_acc_val,
                        top5_acc_val,
                        duration_val)]),
                os.getenv('PBS_JOBID').split('.')[0],
                kwargs['email']))
        return
