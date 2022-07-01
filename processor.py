import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
import math

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
        model: nn.Module,
        num_classes: int):
        """
        Args:
            model : ``torch.nn.Module``
                Configured PyTorch model.
            
            num_classes : ``int``
                Number of action classification classes.
        """

        self.model = model
        self.ce = nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')
        self.num_classes = num_classes


    def train(
        self, 
        save_dir: str, 
        dataloader: DataLoader,
        device: torch.device,
        epochs: int,
        checkpoints: list[int],
        learning_rate: float,
        **kwargs):
        """Trains the model, given user-defined training parameters.

        TODO:
            ``1.`` Provide useful prediction statistics (e.g. IoU, jitter, etc.).
        """

        # set layers to training mode if behavior of any differs between train and prediction
        # (prepares Dropout and BatchNormalization layers to disable and to learn parameters, respectively)
        self.model.train()

        # move the model to the compute device if available (CPU, GPU, TPU, etc.)
        self.model.to(device)
        self.base_lr = learning_rate

        # setup the optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.base_lr)

        # train the model for num_epochs
        # (dataloader is automatically shuffled after each epoch)
        for epoch in range(epochs):            
            epoch_loss = 0
            correct = 0
            total = 0

            # sweep through the training dataset in minibatches
            for captures, labels in dataloader:
                # move both data to the compute device
                # (captures is a batch of full-length captures, label is a batch of ground truths)
                captures = captures[:,:,:,:,kwargs['subject']].to(device)
                # broadcast the labels across the capture length dimension for framewise comparison to predictions
                # expanding labels tensor does not allocate new memory, only creates a new view on existing tensor
                labels = labels[:,None].expand(-1,captures.shape[2]).to(device)
                
                # print('moved data to the device')
                # zero the gradient buffers
                optimizer.zero_grad()
                
                # make predictions and compute the loss
                # supply the captures in user-defined length (realtime, buffered, batch)
                # (tensor will have identical number of dimensions, but only 2nd dimension will change 1 -> L)
                predictions = []

                for minibatch in range(math.ceil(captures.shape[2]/kwargs['buffer'])):
                    # forward pass the minibatch through the model for the corresponding subject
                    # the input tensor has shape (N, C, L, V): N-batch, C-channels, L-length, V-nodes
                    predictions.append(self.model(captures[:,:,minibatch*kwargs['buffer']:(minibatch+1)*kwargs['buffer']]))
                
                # stack the predictions back into the dimension of the input capture
                # (N, C, L'): N-batch, C-classes, L'-full length
                output = torch.cat(predictions, dim=2)

                # cross-entropy expects output as class indices (N, C, K), with labels (N, K): 
                # N-batch, C-class, K-extra dimension (capture length)
                loss = self.ce(output, labels)
                epoch_loss += loss.data.item()

                # backward pass to compute the gradients
                loss.backward()

                # update parameters based on the computed gradients
                optimizer.step()

                # calculate the predictions statistics
                # this only sums the number of correctly predicted frames, but doesn't look at prediction jitter
                _, predicted = torch.max(output, 1)
                correct += torch.sum(predicted == labels).data.item()
                total += labels.numel()

            # checkpoint the model during training at specified epochs
            if epoch in checkpoints:
                torch.save(self.model.state_dict(), "{0}/epoch-{1}.model".format(save_dir, epoch + 1))
                torch.save(optimizer.state_dict(), "{0}/epoch-{1}.opt".format(save_dir, epoch + 1))
            
            print("[epoch {0}]: epoch loss = {1}, acc = {2}".format(
                epoch + 1, 
                epoch_loss / len(dataloader),
                float(correct) / total), flush=True, file=kwargs['log'][1])

        # save the final model
        torch.save(self.model.state_dict(), "{0}/final.model".format(save_dir))
        torch.save(optimizer.state_dict(), "{0}/final.opt".format(save_dir))
        return


    # def predict(
    #     self, 
    #     model_dir, 
    #     results_dir, 
    #     features_path, 
    #     vid_list_file, 
    #     epoch, 
    #     actions_dict, 
    #     device, 
    #     sample_rate):
    #     """Performs only the forward pass for inference.
    #     """
        
    #     self.model.eval()
        
    #     with torch.no_grad():
    #         self.model.to(device)
    #         self.model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))
    #         file_ptr = open(vid_list_file, 'r')
    #         list_of_vids = file_ptr.read().split('\n')[:-1]
    #         file_ptr.close()
    #         for vid in list_of_vids:
    #             string2 = vid[:-10]
    #             features = np.load(features_path + string2 + 'input' + '.npy')
    #             features = get_features(features)
    #             features = features[:, ::sample_rate, :, :]
    #             input_x = torch.tensor(features, dtype=torch.float)
    #             input_x.unsqueeze_(0)
    #             N, C, T, V, M = input_x.size()
    #             input_x = input_x.to(device)
    #             predictions = self.model(input_x, torch.ones(N,2,T).to(device))
    #             _, predicted = torch.max(predictions[-1].data, 1)
    #             predicted = predicted.squeeze().data.detach().cpu().numpy()
    #             recognition = []
    #             for i in range(len(predicted)):
    #                 recognition = np.concatenate((recognition, [list(actions_dict.keys())[list(actions_dict.values()).index(predicted[i].item())]]*sample_rate))
    #             f_name = vid[:-4]
    #             f_ptr = open(results_dir + "/" + f_name, "w")
    #             f_ptr.write("### Frame level recognition: ###\n")
    #             f_ptr.write(' '.join(recognition))
    #             f_ptr.close()

    #     ##################################################
    #     # TODO:
    #     ##################################################
    #     with torch.no_grad():
    #         for data in testloader:
    #             images, labels = data
    #             outputs = net(images)
    #             _, predictions = torch.max(outputs, 1)
    #             # collect the correct predictions for each class
    #             for label, prediction in zip(labels, predictions):
    #                 if label == prediction:
    #                     correct_pred[classes[label]] += 1
    #                 total_pred[classes[label]] += 1


    #     # print accuracy for each class
    #     for classname, correct_count in correct_pred.items():
    #         accuracy = 100 * float(correct_count) / total_pred[classname]
    #         print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
