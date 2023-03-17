import torch
import exp_utils, utils
import argparse

def main(dataset):

    batch_size = 64
    val_size = 5000
    train_size = 45000

    if dataset == 'CIFAR10':
        num_classes = 10
        _, __, testloader = utils.get_CIFAR10(batch_size, val_size, train_size)
    elif dataset == 'CIFAR100':
        num_classes = 100
        _, __, testloader = utils.get_CIFAR100(batch_size, val_size, train_size)
    else:
        raise ValueError('Dataset not supported.')

    device = torch.device("cuda:1")

    seeds = [0,1,2]
    depth = 6
    #n_blocks_small = 12
    patch_size = 4
    max_len = 100
    embed_dim = 512
    layers = 6
    channels = 3
    heads = 16

    model_name = 'ViT'

    normal_base_paths = [f'saved_models/{dataset}/Centralized{model_name}{depth}_{i}/' for i in seeds]
    adaptable_base_paths = [f'saved_models/{dataset}/CentralizedAccordion{model_name}{depth}_{i}/' for i in seeds]
    #distill_base_paths = [f'saved_models/{dataset}/CentralizedDistillAccordion{model_name}{depth}_{i}/' for i in seeds]
    #small_base_paths = [f'saved_models/{dataset}/CentralizedSmall{model_name}{depth}_{n_blocks_small}_{i}/' for i in seeds]

    if model_name == 'ResNet':
        exp_utils.get_results(normal_base_paths, testloader, seeds, depth, num_classes, device)
        exp_utils.get_results(adaptable_base_paths, testloader, seeds, depth, num_classes, device)
        #exp_utils.get_results(distill_base_paths, testloader, seeds, depth, num_classes, device)
        #exp_utils.get_results(small_base_paths, testloader, seeds, depth, num_classes, device)
    elif model_name == 'ViT':
        exp_utils.get_transformer_results(normal_base_paths, testloader, seeds, device, 
                                          patch_size = patch_size,
                                          max_len=max_len,
                                          embed_dim=embed_dim,
                                          classes = num_classes,
                                          layers = layers,
                                          channels = channels,
                                          heads = heads)
        exp_utils.get_transformer_results(adaptable_base_paths, testloader, seeds, device, 
                                          patch_size = patch_size,
                                          max_len=max_len,
                                          embed_dim=embed_dim,
                                          classes = num_classes,
                                          layers = layers,
                                          channels = channels,
                                          heads = heads)
        '''
        exp_utils.get_transformer_results(distill_base_paths, testloader, seeds, device, 
                                          patch_size = patch_size,
                                          max_len=max_len,
                                          embed_dim=embed_dim,
                                          classes = num_classes,
                                          layers = layers,
                                          channels = channels,
                                          heads = heads)
        '''

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog = "CentralizedExperiments") 
    parser.add_argument('dataset')
    args = parser.parse_args()

    main(args.dataset)