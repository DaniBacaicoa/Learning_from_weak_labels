"""" Load Datasets for classification problems
    Authors: Daniel Bacaicoa-Barber, June 2022
             Jesús Cid-Sueiro (Original code)
             Miquel Perelló-Nieto (Original code)
"""
#improting standard libraries
import numpy as np

import sklearn.datasets as skd
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

#importing database library
import openml

# importing torch related libraries
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import datasets, transforms

#importing local methods
from utils.utils_weakener import binarize_labels


class OpenML_Dataset(Dataset):
    '''
    The dataloader returns a pytorch dataset
    inputs:
        dataset: [str/int] refers to the selected dataset
            Synthetic datasets
            - 'hypercube'
            - 'blobs'
            - 'blobs2'
            Sklearn's datasets
            - 'iris'
            - 'digits'
            - 'covtype'
            Openml edited datasets
            - 'glass_2' (glass dataset but with 3 classes)
            Openml datasets
            -   'iris': 61, 'pendigits': 32, 'glass': 41, 'segment': 36,
                'vehicle': 54, 'vowel': 307, 'wine': 187, 'abalone': 1557,
                'balance-scale': 11, 'car': 21, 'ecoli': 39, 'satimage': 182,
                'collins': 478, 'cardiotocography': 1466, 'JapaneseVowels': 375,
                'autoUniv-au6-1000': 1555, 'autoUniv-au6-750': 1549,
                'analcatdata_dmft': 469, 'autoUniv-au7-1100': 1552,
                'GesturePhaseSegmentationProcessed': 4538,
                'autoUniv-au7-500': 1554, 'mfeat-zernike': 22, 'zoo': 62,
                'page-blocks': 30, 'yeast': 181, 'flags': 285,
                'visualizing_livestock': 685, 'diggle_table_a2': 694,
                'prnn_fglass': 952, 'confidence': 468, 'fl2000': 477,
                'blood-transfusion': 1464, ' banknote-authentication': 1462
            UCI dtasets
            - TBD
    '''

    def __init__(self, dataset, train_size = 0.7, test_size = None, valid_size = None, batch_size = 64, shuffling = False, splitting_seed = 47):

        self.dataset = dataset

    
        self.tr_size = train_size # It can be a two or three size tuple (tr_s,val_s,test_s)
        self.val_size = valid_size
        self.test_size = test_size

        self.weak_labels = None
        self.virtual_labels = None

        self.batch_size = batch_size
        
        self.shuffle = shuffling
        
        self.splitting_seed = splitting_seed

        openml_ids = {'iris': 61, 'pendigits': 32, 'glass': 41, 'segment': 36,
                      'vehicle': 54, 'vowel': 307, 'wine': 187, 'abalone': 1557,
                      'balance-scale': 11, 'car': 21, 'ecoli': 39, 'satimage': 182,
                      'collins': 478, 'cardiotocography': 1466, 'JapaneseVowels': 375,
                      'autoUniv-au6-1000': 1555, 'autoUniv-au6-750': 1549,
                      'analcatdata_dmft': 469, 'autoUniv-au7-1100': 1552,
                      'GesturePhaseSegmentationProcessed': 4538,
                      'autoUniv-au7-500': 1554, 'mfeat-zernike': 22, 'zoo': 62,
                      'page-blocks': 30, 'yeast': 181, 'flags': 285,
                      'visualizing_livestock': 685, 'diggle_table_a2': 694,
                      'prnn_fglass': 952, 'confidence': 468, 'fl2000': 477,
                      'blood-transfusion': 1464, ' banknote-authentication': 1462}
        if self.dataset in openml_ids:
            data = openml.datasets.get_dataset(openml_ids[self.dataset])
            X, y, categorical, feature_names = data.get_data(
                target=data.default_target_attribute,
            )
            X = X.values
            le = preprocessing.LabelEncoder()
            y = le.fit_transform(y) #Tis encodes labels into classes [0,1,2,...,n_classes-1]
            X, y = shuffle(X, y, random_state = self.splitting_seed)
        elif isinstance(self.dataset, int):
            data = openml.datasets.get_dataset(self.dataset)
            X, y, categorical, feature_names = data.get_data(
                target=data.default_target_attribute,
            )
            X = X.values
            le = preprocessing.LabelEncoder()
            y = le.fit_transform(y)
            X, y = shuffle(X, y)
        elif self.dataset == 'hypercube':
            X, y = skd.make_classification(
                n_samples=400, n_features=40, n_informative=40,
                n_redundant=0, n_repeated=0, n_classes=4,
                n_clusters_per_class=2,
                weights=None, flip_y=0.0001, class_sep=1.0, hypercube=True,
                shift=0.0, scale=1.0, shuffle=True, random_state=None)
        elif self.dataset == 'blobs':
            X, y = skd.make_blobs(
                n_samples=400, n_features=2, centers=20, cluster_std=2,
                center_box=(-10.0, 10.0), shuffle=True, random_state=None)
        elif self.dataset == 'blobs2':
            X, y = skd.make_blobs(
                n_samples=400, n_features=4, centers=10, cluster_std=1,
                center_box=(-10.0, 10.0), shuffle=True, random_state=None)
        elif self.dataset == 'iris':
            dataset = skd.load_iris()
            X = dataset.data
            y = dataset.target
        elif self.dataset == 'digits':
            dataset = skd.load_digits()
            X = dataset.data
            y = dataset.target
        elif self.dataset == 'covtype':
            dataset = skd.fetch_covtype()
            X = dataset.data
            y = dataset.target - 1
        elif self.dataset == 'glass_2':

            dataset = openml.datasets.get_dataset(41)
            X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
            class_mask = (y == 'build wind float') | (y == 'build wind non-float') | (y == 'headlamps')
            X_filtered, y_filtered = X[class_mask], y[class_mask]
            le = preprocessing.LabelEncoder()
            y_filtered = le.fit_transform(y_filtered)
            X, y = shuffle(X_filtered, y_filtered, random_state=self.splitting_seed)
            X = X.values
        else:
            print('TBD. Sorry for the inconvenience.')

        self.num_classes = len(np.unique(y))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size = self.tr_size, random_state = self.splitting_seed)
        X_train = torch.from_numpy(X_train).to(torch.float32)
        X_test = torch.from_numpy(X_test).to(torch.float32)
        y_train = torch.from_numpy(y_train).to(torch.long)
        y_test = torch.from_numpy(y_test).to(torch.long)

        self.train_dataset = TensorDataset(X_train, y_train)
        self.test_dataset = TensorDataset(X_test, y_test)

        # This is done to mantain coherence between de datset classes
        self.train_dataset.data = self.train_dataset.tensors[0]
        self.train_dataset.targets = self.train_dataset.tensors[1]
        self.test_dataset.data = self.test_dataset.tensors[0]
        self.test_dataset.targets = self.test_dataset.tensors[1]

        self.train_num_samples = self.train_dataset.data.shape[0]
        self.test_num_samples = self.test_dataset.data.shape[0]

        self.num_features = self.train_dataset.data.shape[1]

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
            indices = torch.Tensor(list(range(len(self.train_dataset)))).to(torch.long)
        #print(indices)
        if self.weak_labels is None:
            tr_dataset = TensorDataset(self.train_dataset.data[indices],
                                                     self.train_dataset.targets[indices])
        else:
            tr_dataset = TensorDataset(self.train_dataset.data[indices], self.weak_labels[indices],
                                                     self.train_dataset.targets[indices], indices)

        self.train_loader = DataLoader(tr_dataset, batch_size=self.batch_size, shuffle=self.shuffle,
                                                        num_workers=0)
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

