import torch.nn as nn
import torch.optim as optim
import torch
import utils
import train
import numpy as np
import time
from ptflops import get_model_complexity_info
import json
import pickle5 as pickle
from models.AdaptableViT import AdaptableVisionTransformer

def get_val_accuracies(base_paths, seeds):
    val_accuracies = []
    for i in seeds:
        with open(base_paths[i] + 'val_acc.pkl', 'rb') as f:
            val_acc= pickle.load(f)
            val_accuracies.append(val_acc)

    return val_accuracies

def get_accuracies(net, dloader, criterion, device):

    net.eval()
    accuracies = []

    net.reconfigure_blocks(net.max_blocks)
    active_blocks = net.active_blocks

    for num_blocks in range(active_blocks + 1):
        print(f"Progress: {num_blocks}/{active_blocks}", end = '\r')

        net.reconfigure_blocks(num_blocks)
        loss, accuracy = train.loss_and_accuracy(net, dloader, criterion, device)
        accuracies.append(accuracy)
        
    print('')
    return accuracies

def get_results(base_paths, testloader, seeds, depth, num_classes, device):
    
    model_paths = [path + 'model' for path in base_paths]
    nets = [utils.load_model(model_path, depth, num_classes, device) for model_path in model_paths]
    model_fractions = utils.get_model_fractions(nets[0])
    criterion = nn.CrossEntropyLoss()
    accuracies = [get_accuracies(net, testloader, criterion, device) for net in nets]

    for i in seeds:

        results = dict()
        results['model_fractions'] = model_fractions
        results['accuracies'] = list(accuracies[i])

        with open(base_paths[i] + "results.json", 'w') as f:
            json.dump(results, f, indent=4)

def get_transformer_results(base_paths, testloader, seeds, device, **kwargs):
    
    model_paths = [path + 'model' for path in base_paths]
    nets = [utils.load_transformer(model_path, device, **kwargs) for model_path in model_paths]
    model_fractions = utils.get_model_fractions(nets[0])
    criterion = nn.CrossEntropyLoss()
    accuracies = [get_accuracies(net, testloader, criterion, device) for net in nets]

    for i in seeds:

        results = dict()
        results['model_fractions'] = model_fractions
        results['accuracies'] = list(accuracies[i])

        with open(base_paths[i] + "results.json", 'w') as f:
            json.dump(results, f, indent=4)


def get_fractional_macs(net, model_input, block_inputs):
    
    net_macs, params = get_model_complexity_info(net, model_input, as_strings=False,
                                            print_per_layer_stat=False, verbose=False)
                                            
    print(f"The model has {params} parameters and full model is {net_macs} Mac")

    count = 0
    block_macs = []

    for i, layer in enumerate(net.layers):
        for j, block in enumerate(layer):
            insize = block_inputs[count]
            macs, params = get_model_complexity_info(block, insize, as_strings=False,
                                            print_per_layer_stat=False, verbose=False)
            print(f"Layer {i} block {j} has {params} params and {macs} Mac")
            block_macs.append(macs)
            count += 1

    fractional_macs = [net_macs]
    for i in range(len(block_macs)):
        fractional_macs = [fractional_macs[0] - block_macs[-i-1]]+ fractional_macs

    return block_macs, fractional_macs


def get_macs(net, insize):
    macs, _ = get_model_complexity_info(net, insize, as_strings=False,
                                    print_per_layer_stat=False, verbose=False)
    print(f"Model has {macs:e} MAC")
    return macs

def forward_time(net, epochs, dloader, device):

    times = []

    for epoch in range(epochs):
        for data in dloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            start = time.time()
            net(images)
            end = time.time()

            times.append(end - start)

    return np.sum(times) / epochs

def get_forward_times(net, dloader, device):

    net.eval()
    times = []

    net.reconfigure(1.)
    active_blocks = net.active_blocks

    # run one epoch to avoid the time data is being loaded
    forward_time(net, epochs=1, dloader=dloader, device=device)

    for num_blocks in range(active_blocks+1):
        print(f"Progress: {num_blocks}/{active_blocks}", end = '\r')

        net.reconfigure_blocks(num_blocks)
        times.append(forward_time(net, epochs = 5, dloader=dloader, device=device))
        print(f"Model fraction : {net.model_fraction}, forward time: {times[-1]}")

    return times

def epoch_time(net, epochs, dloader, device):

    times = []
    optimizer = optim.SGD(net.parameters(), 0.01, 0.9, 1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for data in dloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            start = time.time()

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()  

            end = time.time()

            times.append(end - start)

    return np.sum(times) / epochs

def get_epoch_times(net, dloader, device):

    net.eval()
    times = []

    net.reconfigure(1.)
    active_blocks = net.active_blocks

    # run one epoch to avoid the time data is being loaded
    epoch_time(net, epochs=1, dloader=dloader, device = device)

    for num_blocks in range(active_blocks+1):
        print(f"Progress: {num_blocks}/{active_blocks}", end = '\r')

        net.reconfigure_blocks(num_blocks)
        times.append(epoch_time(net, epochs = 5, dloader=dloader, device = device))
        print(f"Model fraction : {net.model_fraction}, epoch time: {times[-1]}")

    return times