import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import utils
import train_transformer
from fed_data import FedDataLoader, FedDataSet

class AdaptableFedTrainer:

    def __init__(self, global_model, trainset, valset, n_devices, num_blocks, n_device_per_round, device, 
                global_epochs = 60, patience = 8, factor = 0.1, scheduling = True, local_epochs = 10, 
                lr = 0.05, momentum = 0.9, weight_decay = 1e-4, seed = 0, testset = None, 
                device_num_blocks = None, distillation = False, beta = 0):
        
        # seed for reproducibility
        self.seed = seed
        self.n_devices = n_devices
        self.fed_dataset = FedDataSet(trainset, valset, n_devices, seed = self.seed)
        self.fed_dataloader = FedDataLoader(n_devices, self.fed_dataset, seed = self.seed, batch_sizes=[32]*n_devices)   

        # test set is not used anywhere in training, only to track how test accuracy changes with epochs
        if testset is None:
            dset, self.testset = utils.get_CIFAR10_data()
            del dset
            self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=len(self.testset),
                                                    shuffle=False, num_workers=1) 
        else:
            self.testset = testset
            self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=len(self.testset),
                                                    shuffle=False, num_workers=1)

        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.patience = patience
        self.factor = factor
        self.scheduling = scheduling
        self.distillation = distillation
        self.beta = beta

        self.num_blocks = num_blocks
        self.global_model = global_model
        self.n_device_per_round = n_device_per_round
        self.global_epochs = global_epochs
        self.local_epochs = local_epochs
        self.device = device

        self.local_models = []
        self.optimizers = []
        self.schedulers = []
        for i in range(n_devices):  
            self.local_models.append(copy.deepcopy(global_model))
            self.local_models[i].reconfigure_blocks(self.num_blocks[i])
            self.local_models[i].to(device)

            self.optimizers.append(optim.SGD(self.local_models[i].parameters(), self.lr, 
                                            self.momentum, self.weight_decay))
            self.schedulers.append(optim.lr_scheduler.ReduceLROnPlateau(self.optimizers[i], 'max', patience = patience, 
                                                            factor = factor, verbose = True, min_lr=2e-4))
                       
        self.criterion = nn.CrossEntropyLoss()
        
        if device_num_blocks is None:
            device_num_blocks = num_blocks
        
        self.device_num_blocks = sorted(device_num_blocks, reverse=True)

        self.global_epoch_losses = []
        self.global_val_accuracies = []
        self.global_val_losses = []
        self.global_test_accuracies = []


    def train_one_epoch(self, distill = False):

        self.global_model.train()

        device_ids = np.random.choice(self.n_devices, self.n_device_per_round, replace=False)

        global_epoch_loss = 0

        for i,idx in enumerate(device_ids):
            print(f"Device progress: {i + 1}/{len(device_ids)}", end = '\r')
            global_epoch_loss += self.train_one_device(idx, distill)
        
        print("")

        test_acc = self.get_device_accuracies(self.device_num_blocks)

        print(f"Client Accuracies: {test_acc}")
        print(f"Client Average Test Accuracy: {np.mean(test_acc):.3f}")
        self.global_test_accuracies.append(test_acc)

        global_weights = self.average_model_weights(device_ids)
        self.global_model.load_state_dict(global_weights)

        for i in range(self.n_devices):
            self.local_models[i].load_state_dict(global_weights)

        return global_epoch_loss

    def train_one_device(self, idx, distill = False):

        local_model = self.local_models[idx]
        local_model.train()
        trainloader = self.fed_dataloader.get_trainloader(idx)
        optimizer = self.optimizers[idx]

        average_local_epoch_loss = 0
        average_dist_loss = 0

        for epoch in range(self.local_epochs):

            local_epoch_loss = 0
            local_epoch_loss1 = 0
            local_epoch_loss2 = 0
            num_batches = 0

            for i, data in enumerate(trainloader, 0):

                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # zero the parameter gradients
                optimizer.zero_grad()

                if distill and local_model.active_blocks != self.device_num_blocks[-1]:
                    active_blocks = local_model.active_blocks
                    next_device_blocks = self.get_next_device_blocks(active_blocks)

                    outputs, curr_activation = train_transformer.forward_and_final_activation(local_model, inputs, eval = False) 

                    local_model.reconfigure_blocks(next_device_blocks)
                    _, next_activation =  train_transformer.forward_and_final_activation(local_model, inputs, eval = False) 
                    local_model.reconfigure_blocks(active_blocks)

                    c = nn.MSELoss(reduction='mean')

                    loss1 = self.criterion(outputs, labels)
                    loss2 = torch.mul(c(curr_activation, next_activation), self.beta)
                    loss = loss1 + loss2

                else:

                    # forward + backward + optimize
                    outputs = local_model(inputs)
                    loss1 = self.criterion(outputs, labels)
                    loss2 = 0

                    loss = loss1 + loss2
                
                loss.backward()

                if distill:
                    #nn.utils.clip_grad_value_(parameters=local_model.parameters(), clip_value=1e-1)
                    pass

                optimizer.step()  

                local_epoch_loss += loss.item()
                local_epoch_loss1 += loss1.item()
                local_epoch_loss2 += loss2
                num_batches += 1

            device_loss = local_epoch_loss / num_batches
            average_local_epoch_loss += device_loss
            average_dist_loss += local_epoch_loss2 / num_batches
        
        average_local_epoch_loss /= self.local_epochs
        #average_local_dist_loss = average_dist_loss / self.local_epochs

        return average_local_epoch_loss


    def train(self):

        print("Started Training!")

        for epoch in range(self.global_epochs):

            print("")
            print(f"Epoch = {epoch + 1}/{self.global_epochs}")

            warmup_before_distillation = 5
            distill = self.distillation and epoch > warmup_before_distillation
            global_epoch_loss = self.train_one_epoch(distill)
            global_epoch_loss /= self.n_device_per_round
            self.global_epoch_losses.append(global_epoch_loss)

            print(f"Loss = {global_epoch_loss:.3f}")

            # calculate validation accuracy and loss across all devices
            global_val_loss, global_val_accuracy = self.loss_and_accuracy()
            self.global_val_accuracies.append(global_val_accuracy)
            self.global_val_losses.append(global_val_loss)

            for i in range(self.n_devices):
                if self.scheduling:
                    self.schedulers[i].step(global_val_accuracy)

            print(f"Validation Loss = {global_val_loss:.3f}, Validation Accuracy = {global_val_accuracy:.3f}")
        
        print("Finished Training!")


    def loss_and_accuracy(self):
        # loss and accuracy on the validation set

        self.global_model.eval()
        valloader = self.fed_dataset.get_valloader()

        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in valloader:
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)
                # calculate outputs by running images through the network
                outputs = self.global_model(images)
                loss = self.criterion(outputs, labels)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total

        return loss, accuracy

    def average_model_weights(self, device_ids):

        weights = copy.deepcopy(self.local_models[device_ids[0]].state_dict())
        for key in weights.keys():
            for i in device_ids[1:]:
                weights[key] += self.local_models[i].state_dict()[key]
            weights[key] = torch.div(weights[key],len(device_ids))
        
        return weights

    def get_device_accuracies(self, num_blocks):

        acc_dict = dict()
        acc = []

        active_blocks = self.global_model.active_blocks

        for i in num_blocks:
            self.global_model.reconfigure_blocks(i)
            if i in acc_dict:
                test_accuracy = acc_dict[i]
            else:
                _, test_accuracy = train_transformer.loss_and_accuracy(self.global_model, self.testloader, self.criterion, self.device)
                acc_dict[i] = test_accuracy
            acc.append(test_accuracy)
        
        self.global_model.reconfigure_blocks(active_blocks)
        
        return acc

    def get_next_device_blocks(self, active_blocks):

        i = 0
        while self.device_num_blocks[i] >= active_blocks:
            i += 1
        
        return self.device_num_blocks[i+1]