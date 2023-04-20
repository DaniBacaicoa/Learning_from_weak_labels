"""" Load Datasets for comparing with Valen
    Authors: Daniel Bacaicoa-Barber, June 2022
             Jesús Cid-Sueiro (Original code)
             Miquel Perelló-Nieto (Original code)
"""

#importing torch related libraries
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import datasets, transforms

#importing local methods
from utils.utils_weakener import binarize_labels



class Torch_Dataset(Dataset):
    '''
    Bk_Dataset returns a dataset comparable with those in Valen.
    It incorporates methods for getting the train/test feature/label sets
        as of producing a dataloader (a.k.a. trainloader and testloader)

    inputs:
        ds_id: [str] refers to the selected dataset
            - 'mnist'
            - 'mnist'
            - 'mnist'
            - [TBD] 'cifar10'
        shuffle (opt): [bool] Whether shffling the ds or not
        seed (opt): [int] The seed for replicating the shuffle process
    '''

    def __init__(self, dataset='mnist', batch_size=64, shuffle=True):
        self.dataset = dataset.upper()
        self.weak_labels = None
        self.batch_size = batch_size
        self.shuffle = shuffle

        if self.dataset == 'MNIST':
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        elif self.dataset == 'KMNIST':
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1904,), (0.3475,))
            ])
        elif self.dataset == 'FMNIST':
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,))
            ])
        # [TBD] Include Cifar10 dataset

        self.train_dataset = datasets.__dict__[self.dataset](root='./data', train=True, transform=self.transform,
                                                             download=True)
        self.train_dataset.data = self.train_dataset.data.to(torch.float32)
        self.test_dataset = datasets.__dict__[self.dataset](root='./data', train=False, transform=self.transform,
                                                            download=True)
        self.test_dataset.data = self.test_dataset.data.to(torch.float32)
        
        self.num_classes = torch.max(torch.unique(self.train_dataset.targets))+1
        self.train_num_samples = self.train_dataset.data.shape[0]
        self.test_num_samples = self.test_dataset.data.shape[0]

        #Flattening the images
        self.train_dataset.data = self.train_dataset.data.view((self.train_num_samples,-1))
        self.test_dataset.data = self.test_dataset.data.view((self.test_num_samples,-1))

        self.num_features = self.train_dataset.data.shape[1]

        #One hot encoding of the labels
        self.train_dataset.targets = binarize_labels(self.num_classes, self.train_dataset.targets)
        self.test_dataset.targets = binarize_labels(self.num_classes, self.test_dataset.targets)


    def __getitem__(self, index):
        if self.weak_labels is None:
            x = self.train_dataset.data[index]
            y = self.train_dataset.targets[index]
            return x, y
        else:
            x = self.train_dataset.data[index]
            w = self.weak_labels[index]
            y = self.train_dataset.targets[index]
            return x, w, y

    def get_dataloader(self, indices=None):
        if indices is None:
            #indices = torch.Tensor(list(range(len(self.train_dataset)))).to(torch.long)
            indices = torch.arange(len(self.train_dataset))
        if self.weak_labels is None:
            dataset = TensorDataset(self.train_dataset.data[indices], self.train_dataset.targets[indices])
        else:
            dataset = TensorDataset(self.train_dataset.data[indices], self.weak_labels[indices], self.train_dataset.targets[indices],indices)

        self.train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=0)
        self.test_loader = DataLoader(TensorDataset(
            self.test_dataset.data, self.test_dataset.targets
        ), batch_size=self.batch_size, shuffle=self.shuffle, num_workers=0)
        return self.train_loader, self.test_loader

    def get_data(self):
        train_x = self.train_dataset.data
        train_y = self.train_dataset.targets
        test_x = self.test_dataset.data
        test_y = self.test_dataset.targets

        return train_x, train_y, test_x, test_y

    def include_weak(self, z):
        if torch.is_tensor(z):
            self.weak_labels = z
        else:
            self.weak_labels = torch.from_numpy(z)

