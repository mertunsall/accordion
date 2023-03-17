import torch
import numpy as np
import random

# allocates the dataset across different devices
class FedDataSet:

    def __init__(self, trainset, valset, n_devices, train_nums = None, val_nums = None, seed = 0):

        self.trainset = trainset
        self.valset = valset
        self.n_devices = n_devices

        self.lenvalset = len(valset)
        self.lentrainset = len(trainset)

        self.seed = seed

        self._allocate_data(train_nums, val_nums)
    
    def _allocate_data(self, train_nums = None, val_nums = None):
        
        if train_nums is None:
            trainperdevice = len(self.trainset) // self.n_devices
            self.train_nums = [trainperdevice]*self.n_devices
        else:
            self.train_nums = train_nums

        if val_nums is None:
            valperdevice = len(self.valset) // self.n_devices
            self.val_nums = [valperdevice]*self.n_devices
        else:
            self.val_nums = val_nums

        self.trainsets = list(torch.utils.data.random_split(self.trainset, self.train_nums))
        self.valsets = list(torch.utils.data.random_split(self.valset, self.val_nums))

    def get_trainset(self, device_id):
        
        if device_id >= self.n_devices or device_id < 0:
            raise ValueError(f"Invalid device_id: {device_id}")

        return self.trainsets[device_id]
    
    def get_valset(self, device_id):

        if device_id >= self.n_devices or device_id < 0:
            raise ValueError(f"Invalid device_id: {device_id}")

        return self.valsets[device_id]

    def get_valloader(self, batch_size = 64):

        g = torch.Generator()
        g.manual_seed(self.seed)

        return torch.utils.data.DataLoader(self.valset, batch_size=batch_size,
                                    shuffle = False, num_workers=1, 
                                    worker_init_fn = seed_worker, generator = g)

class FedDataLoader:

    def __init__(self, n_devices, fed_dataset, batch_sizes = None, seed = None):

        self.n_devices = n_devices
        self.fed_dataset = fed_dataset       

        if seed is None:
            self.seed = 0
        else:
            self.seed = seed

        self._create_dataloaders(batch_sizes)
    
    def _create_dataloaders(self, batch_sizes = None):

        if batch_sizes is None:
            self.batch_sizes = [64]*self.n_devices
        elif isinstance(batch_sizes, int):
            self.batch_sizes = [batch_sizes]*self.n_devices
        elif isinstance(batch_sizes, list):
            if len(batch_sizes) != self.n_devices:
                raise ValueError("The array batch_sizes has to have n_devices number of elements.")
            self.batch_sizes = batch_sizes            

        self.trainloaders = []
        self.valloaders = []
        self.generators = []

        for i in range(self.n_devices):

            self.generators.append(torch.Generator())
            self.generators[-1].manual_seed(self.seed)

            trainset_i = self.fed_dataset.get_trainset(i)
            self.trainloaders.append(torch.utils.data.DataLoader(trainset_i, batch_size=self.batch_sizes[i],
                                    shuffle = True, num_workers=1, 
                                    worker_init_fn = seed_worker, generator = self.generators[-1]))

            self.generators.append(torch.Generator())
            self.generators[-1].manual_seed(self.seed)
            
            valset_i = self.fed_dataset.get_valset(i)
            self.valloaders.append(torch.utils.data.DataLoader(valset_i, batch_size=self.batch_sizes[i],
                                    shuffle = True, num_workers=1, 
                                    worker_init_fn = seed_worker, generator = self.generators[-1]))

    def get_trainloader(self, device_id):
        
        if device_id >= self.n_devices or device_id < 0:
            raise ValueError(f"Invalid device_id: {device_id}")

        return self.trainloaders[device_id]
    
    def get_valloader(self, device_id):

        if device_id >= self.n_devices or device_id < 0:
            raise ValueError(f"Invalid device_id: {device_id}")

        return self.valloaders[device_id]

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

