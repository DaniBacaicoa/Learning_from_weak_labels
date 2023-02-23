import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

class Bk_Dataset(Dataset):
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
    def __init__(self, dataset = 'mnist', batch_size = 64, shuffle = True):
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
        #[TBD] Include Cifar10 dataset

        self.train_dataset = datasets.__dict__[self.dataset](root='./data', train=True, transform=self.transform, download=True)
        self.test_dataset = datasets.__dict__[self.dataset](root='./data', train=False, transform=self.transform, download=True)

    def get_dataloader(self):
        if self.weak_labels is None:
            self.train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
                self.train_dataset.data, self.train_dataset.targets
            ),batch_size=self.batch_size, shuffle=self.shuffle, num_workers=0)
        else:
            self.train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
                self.train_dataset.data, self.weak_labels, self.train_dataset.targets
            ),batch_size=self.batch_size, shuffle=self.shuffle, num_workers=0)
        self.test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
            self.test_dataset.data, self.test_dataset.targets
        ),batch_size=self.batch_size, shuffle=self.shuffle, num_workers=0)
        return self.train_loader, self.test_loader

    def get_data(self):
        train_x = self.train_dataset.data
        train_y = self.train_dataset.targets
        test_x = self.test_dataset.data
        test_y = self.test_dataset.targets

        return train_x, train_y, test_x, test_y

    def include_weak(self, z):
        self.weak_labels = z