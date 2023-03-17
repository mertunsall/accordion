import torch
import numpy as np
from models.AdaptableResNetCifar import AdaptableResNetCifar
import torch.nn as nn
import torch.optim as optim
import utils
import test

def modify_network(epoch: int, net: AdaptableResNetCifar, model_fractions = None, fraction_probabilities = None, warming_epochs = 15):

    if model_fractions is None or fraction_probabilities is None:
        net.reconfigure(1.)
        return
    
    assert len(model_fractions) == len(fraction_probabilities)

    if epoch <= warming_epochs:
        model_fraction = 1.
    else:
        model_fraction = np.random.choice(model_fractions, p = fraction_probabilities)
        
    net.reconfigure(model_fraction)

def loss_and_accuracy(net, dloader, criterion, device, model_fraction = None):
    # calculate the validation loss and accuracy
    net.eval()
    if model_fraction is not None:
        net.reconfigure(model_fraction)
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in dloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            loss = criterion(outputs, labels)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total

    return loss, accuracy

def adaptable_training(net: AdaptableResNetCifar,
                    criterion, 
                    optimizer, 
                    scheduler, 
                    trainloader, 
                    valloader, 
                    device,
                    fraction_probabilities = None, 
                    n_epochs = 200,
                    distillation = False,
                    beta = 0.1):

    print('Starting training')

    training_loss = []
    training_loss1 = []
    training_loss2 = []
    validation_loss = []
    validation_accuracy = []
    learning_rates = []

    net.train()

    for epoch in range(n_epochs):

        print(f'Epoch: {epoch + 1}/{n_epochs}')

        learning_rates.append(optimizer.param_groups[-1]['lr'])

        running_loss = 0.0
        running_loss1 = 0.0
        running_loss2 = 0.0
        
        num_minibatch = 0
        for i, data in enumerate(trainloader, 0):

            net.train()
            modify_network(epoch, net, fraction_probabilities, warming_epochs=-1)
            #print(model_fractions, fraction_probabilities)
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            if (distillation and net.model_fraction < 1.0) and epoch > 8:

                curr_blocks = net.active_blocks
                net.reconfigure_blocks(net.max_blocks)
                net.eval()
                
                # forward pass with full model
                _, full_activations = forward_and_final_activation(net, inputs)

                net.train()
                net.reconfigure_blocks(curr_blocks)

                outputs, fractional_activations = forward_and_final_activation(net, inputs, eval = False)

                c = nn.MSELoss(reduction='mean')

                #print(f"Active blocks: {net.active_blocks}")

                loss1 = criterion(outputs, labels)
                #print(f"Classification Loss: {loss1}")

                loss2 = torch.mul(c(full_activations, fractional_activations), beta)
                #print(f"Distillation Loss: {loss2}")

                loss = loss1 + loss2
            
            else:
                # forward + backward + optimize
                outputs = net(inputs)

                loss1 = criterion(outputs, labels)
                loss2 = 0

                loss = loss1 + loss2

            loss.backward()

            nn.utils.clip_grad_value_(parameters=net.parameters(), clip_value=1e-1)

            optimizer.step()
            
            running_loss1 += loss1.item()
            running_loss2 += loss2
            running_loss += loss.item()
            num_minibatch += 1
        
        training_loss.append(running_loss / num_minibatch)
        print(f'Training loss: {training_loss[-1]:.3f}')
        training_loss1.append(running_loss1 / num_minibatch)
        print(f'Classification loss: {training_loss1[-1]:.3f}')
        training_loss2.append(running_loss2 / num_minibatch)
        print(f'Distillation loss: {training_loss2[-1]:.3f}')

        net.reconfigure_blocks(net.max_blocks)
        
        val_loss, val_accuracy = loss_and_accuracy(net, valloader, criterion, device)
        validation_loss.append(val_loss)
        validation_accuracy.append(val_accuracy)
        print(f'Accuracy of the network on the {len(valloader.dataset)} validation images: {val_accuracy} %')

        scheduler.step(val_loss)

    print('Finished Training')
    
    return training_loss, validation_loss, validation_accuracy, learning_rates


def train_and_save(net,
         device,
         model_save_path: str, 
         batch_size = 64, 
         val_size = 5000, 
         train_size = 45000,
         lr = 0.05,
         momentum = 0.9,
         weight_decay = 1e-4,
         patience = 15,
         factor = 0.1,
         block_probabilities = None,
         n_epochs = 200,
         dataset = "CIFAR10",
         distillation = False,
         beta = 0.1):

    
    if dataset == "CIFAR10":
        trainloader, valloader, testloader = utils.get_CIFAR10(batch_size, val_size, train_size)
    elif dataset == "CIFAR100":
        trainloader, valloader, testloader = utils.get_CIFAR100(batch_size, val_size, train_size)
    else:
        raise ValueError("The dataset is not defined.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay = weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = patience, factor = factor, verbose = True, min_lr=2e-4)

    tr_loss, val_loss, val_acc, lrates = adaptable_training(net, criterion, optimizer, 
                                                            scheduler, trainloader, valloader, device, 
                                                            block_probabilities, n_epochs, 
                                                            distillation = distillation, beta = beta)
    
    test_loss, test_accuracy = loss_and_accuracy(net, testloader, criterion, device)

    print(f'Accuracy of the network on the {len(testloader.dataset)} test images: {test_accuracy} %')

    torch.save(net.state_dict(), model_save_path)

    return tr_loss, val_loss, val_acc, lrates

def forward_and_final_activation(net: AdaptableResNetCifar, x: torch.Tensor, eval = True):

    activations = []

    def getActivation():
        # the hook signature
        def hook(model, inp, output):
            #print(inp[0].shape)
            output = output.view(output.size(0), -1)
            activations.append(output)
        return hook 

    hook = net.avgpool.register_forward_hook(getActivation())
    
    if eval:
        net.eval()
        
    out = net(x)

    hook.remove()

    return out, activations[0]