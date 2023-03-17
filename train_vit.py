from models.AdaptableViT import AdaptableVisionTransformer
import torch
import utils
import torch.nn as nn
import torch.optim as optim
import train
import numpy as np
from train_transformer import train_and_save


'''

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

device = torch.device("cuda:1")

train_size = 45000
val_size = 5000
batch_size = 64

dset, testset = utils.get_CIFAR10_data()
trainset, valset = torch.utils.data.random_split(dset, [train_size, val_size])
testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset),
                                        shuffle=False, num_workers=1)

trainloader, valloader, testloader = utils.get_CIFAR10(batch_size, val_size, train_size)

patch_size = 4
max_len = 100
embed_dim = 512
num_classes = 10
layers = 12
channels = 3
heads = 16

global_model = AdaptableVisionTransformer(
    patch_size=patch_size,
    max_len=max_len,
    embed_dim=embed_dim,
    classes=num_classes,
    layers=layers,
    channels=channels,
    heads=heads).to(device)

n_epochs = 10
lr = 3e-3

momentum = 0.9

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(global_model.parameters(), lr=lr, momentum=momentum)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr*3, steps_per_epoch=int(np.ceil(train_size/batch_size)), epochs=n_epochs)

print('Starting training')

training_loss = []
training_loss1 = []
training_loss2 = []
validation_loss = []
validation_accuracy = []
learning_rates = []

global_model.train()

for epoch in range(n_epochs):

    print(f'Epoch: {epoch + 1}/{n_epochs}')

    learning_rates.append(optimizer.param_groups[-1]['lr'])

    running_loss = 0.0
    running_loss1 = 0.0
    running_loss2 = 0.0
    
    num_minibatch = 0
    for i, data in enumerate(trainloader, 0):

        global_model.train()
        #modify_network(epoch, net, model_fractions, fraction_probabilities, warming_epochs=-1)
        #print(model_fractions, fraction_probabilities)
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        outputs = global_model(inputs)

        loss1 = criterion(outputs, labels)
        loss2 = 0

        loss = loss1 + loss2

        loss.backward()

        nn.utils.clip_grad_value_(parameters=global_model.parameters(), clip_value=1)

        optimizer.step()
        
        running_loss1 += loss1.item()
        running_loss2 += loss2
        running_loss += loss.item()
        num_minibatch += 1
        scheduler.step()
    
    training_loss.append(running_loss / num_minibatch)
    print(f'Training loss: {training_loss[-1]:.3f}')
    training_loss1.append(running_loss1 / num_minibatch)
    print(f'Classification loss: {training_loss1[-1]:.3f}')
    training_loss2.append(running_loss2 / num_minibatch)
    print(f'Distillation loss: {training_loss2[-1]:.3f}')
    
    val_loss, val_accuracy = loss_and_accuracy(global_model, valloader, criterion, device)
    validation_loss.append(val_loss)
    validation_accuracy.append(val_accuracy)
    print(f'Accuracy of the network on the {len(valloader.dataset)} validation images: {val_accuracy} %')

print('Finished Training')
'''

device = torch.device("cuda:1")

patch_size = 4
max_len = 100
embed_dim = 512
num_classes = 10
layers = 12
channels = 3
heads = 16

global_model = AdaptableVisionTransformer(
    patch_size=patch_size,
    max_len=max_len,
    embed_dim=embed_dim,
    classes=num_classes,
    layers=layers,
    channels=channels,
    heads=heads).to(device)

BASE_PATH =  "saved_models/CIFAR10/CentralizedViT_0/"
SAVE_PATH = BASE_PATH + 'model'


train_and_save(global_model, device, SAVE_PATH, n_epochs = 10)

