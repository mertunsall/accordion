import torch
import torch.nn as nn
import numpy as np
import pickle5 as pickle
import os
import argparse
import json
import random
import time

import utils, train
from fed_trainer import AdaptableFedTrainer
from fed_data import FedDataSet

EXP_TYPES = ['Centralized', 'CentralizedAccordion', 'CentralizedDistillAccordion', 
    'CentralizedSmall', 'Fed', 'FedAccordion', 'FedDistillAccordion', 'FedClass', 'FedSmall']

def save_data(data, name, base_path):
    with open(base_path + f'{name}.pkl', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def run_exp(config):

    print(f"Starting the experiment: {config['name']}")

    SEED = config['seed']
    BASE_PATH = config['base_path']
    SAVE_PATH = config['save_path']

    if not os.path.exists(BASE_PATH):
        os.makedirs(BASE_PATH)

    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    device = torch.device(f"cuda:{config['device_id']}")
    print(f"Training on {device}")

    exp_type = config['exp_type']
    dataset = config['dataset']
    train_size = config['train_size']
    val_size = config['val_size']
    depth = config['depth']
    num_classes = config['num_classes']
    beta = config['beta']
    lr = config['lr']
    n_devices = config['n_devices']
    n_device_per_round = config['n_device_per_round']
    global_epochs = config['global_epochs']
    scheduling = config['scheduling']
    distillation = config['distillation']
    patience = config['patience']
    local_epochs = config['local_epochs']
    n_models = config['n_models']
    confs = config['confs']
    n_device_confs = config['n_device_confs']
    n_blocks = config['n_blocks']
    num_blocks = config['num_blocks']
    device_num_blocks = config['device_num_blocks']
    fraction_probabilities = config['fraction_probabilities']
    train_nums = config['train_nums']
    val_nums = config['val_nums']
    
    # DATASET PREPARATION
    if dataset == "CIFAR10":
        dset, testset = utils.get_CIFAR10_data()
        trainset, valset = torch.utils.data.random_split(dset, [train_size, val_size])
        testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset),
                                                shuffle=False, num_workers=1)

    elif dataset == "CIFAR100":
        dset, testset = utils.get_CIFAR100_data()
        trainset, valset = torch.utils.data.random_split(dset, [train_size, val_size])
        testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset),
                                                shuffle=False, num_workers=1)

    else:
        raise ValueError("This dataset is not supported.")


    if exp_type not in EXP_TYPES:
        raise ValueError("Experiment type not supported.")

    # MODEL CREATION
    if exp_type != "FedClass":

        global_model = utils.get_model(depth, num_classes, device, dataset)

        if exp_type == "FedSmall" or exp_type == "CentralizedSmall":
            global_model.reconfigure_blocks(n_blocks)
    
    elif exp_type == "FedClass":

        global_models = []
        for i in range(n_models):
            global_models.append(utils.get_model(depth, num_classes, device, dataset))
            global_models[i].reconfigure_blocks(confs[i])
            print(f"Model {i} fraction is {global_models[i].model_fraction}")


    # TRAINING    
    if "Centralized" in exp_type:
        print(f"""The {type(global_model)} has {global_model.num_parameters} out of which {global_model.num_adaptable_params} are adaptable.""")
        print(f"""There are {global_model.active_blocks} active blocks and the minimum model fraction is {global_model.min_model_fraction}.""")

        if exp_type in ["Centralized", "CentralizedSmall"]:
            training_loss, val_loss, val_acc, lr_rates = train.train_and_save(global_model, 
                device, SAVE_PATH, val_size=val_size, train_size=train_size, lr = lr,
                patience = patience, n_epochs=global_epochs, dataset=dataset)

        else:
            model_fractions = utils.get_model_fractions(global_model)
            print(f"Fraction Probabilities: {fraction_probabilities}")
            training_loss, val_loss, val_acc, lr_rates = train.train_and_save(global_model, 
                device, SAVE_PATH, train_size=train_size, val_size=val_size,
                model_fractions=model_fractions, fraction_probabilities=fraction_probabilities,
                distillation=distillation, lr = lr, patience=patience, n_epochs=global_epochs, 
                dataset=dataset, beta = beta)

        _, test_accuracy = train.loss_and_accuracy(global_model, testloader, nn.CrossEntropyLoss(), device)

        print(f'Accuracy of the global model on {len(testloader.dataset)} test images: {test_accuracy} %')
            
        save_data(training_loss, 'training_loss', BASE_PATH)
        save_data(val_loss, 'val_loss', BASE_PATH)
        save_data(val_acc, 'val_acc', BASE_PATH)
        save_data(lr_rates, 'learning_rates', BASE_PATH)

    else:
        if exp_type == 'FedClass':

            feddataset = FedDataSet(trainset, valset, n_models, train_nums, val_nums, SEED)

            adaptable_trainers = []
            for i in range(n_models):
                print(f"Training Model {i}")
                tset = feddataset.get_trainset(i)
                vset = feddataset.get_valset(i)
                adaptable_trainers.append(AdaptableFedTrainer(global_models[i], tset, vset, 
                        n_device_confs[i], num_blocks[i], n_device_confs[i], device, 
                        global_epochs, lr = lr, patience = patience, local_epochs = local_epochs, 
                        seed = SEED, testset=testset, scheduling=scheduling))

                start = time.time()
                adaptable_trainers[i].train()
                end = time.time()

                print(f"Training Time for {global_epochs} epochs: {(end-start)/3600:.4f} hours.")

            for i in range(n_models):
                _, test_accuracy = train.loss_and_accuracy(global_models[i], testloader, nn.CrossEntropyLoss(), device)
                print(f'Accuracy of the global model {i} with fraction {global_models[i].model_fraction} on test images: {test_accuracy} %')
                torch.save(global_models[i].state_dict(), SAVE_PATH + str(i))
            
                save_data(adaptable_trainers[i].global_epoch_losses, f'global_epoch_losses{i}', BASE_PATH)
                save_data(adaptable_trainers[i].global_val_accuracies, f'global_val_accuracies{i}', BASE_PATH)
                save_data(adaptable_trainers[i].global_val_losses, f'global_val_losses{i}', BASE_PATH)
                save_data(adaptable_trainers[i].global_test_accuracies, f'global_test_accuracies{i}', BASE_PATH)
            
        else:

            print(f"Training Block distribution: {num_blocks}")
            print(f"Device Block distribution: {device_num_blocks}")

            adaptable_trainer = AdaptableFedTrainer(global_model, trainset, valset, n_devices, num_blocks,
                n_device_per_round, device, global_epochs, patience, local_epochs = local_epochs, lr = lr,
                seed = SEED, testset=testset, scheduling=scheduling, device_num_blocks=device_num_blocks,
                distillation=distillation, beta = beta)

            start = time.time()
            adaptable_trainer.train()
            end = time.time()

            print(f"Training Time for {global_epochs} epochs: {(end-start)/3600:.4f} hours.")

            _, test_accuracy = train.loss_and_accuracy(global_model, testloader, nn.CrossEntropyLoss(), device)

            print(f'Accuracy of the global model on {len(testloader.dataset)} test images: {test_accuracy} %')

            torch.save(global_model.state_dict(), SAVE_PATH)
            
            save_data(adaptable_trainer.global_epoch_losses, 'global_epoch_losses', BASE_PATH)
            save_data(adaptable_trainer.global_val_accuracies, 'global_val_accuracies', BASE_PATH)
            save_data(adaptable_trainer.global_val_losses, 'global_val_losses', BASE_PATH)
            save_data(adaptable_trainer.global_test_accuracies, 'global_test_accuracies', BASE_PATH)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog = "AccordionExperiments") 
    parser.add_argument('config_file')
    args = parser.parse_args()
    config_file = args.config_file

    with open(config_file, 'r') as f:
        config = json.load(f)

    run_exp(config)







