"""" Load Datasets for classification problems
    Authors: Daniel Bacaicoa-Barber, June 2022
             Jesús Cid-Sueiro (Original code)
             Miquel Perelló-Nieto (Original code)
"""
# imprting standard libraries
import numpy as np
import scipy as sp
import itertools
import math
import sklearn
from sklearn import preprocessing, ensemble, datasets
import sklearn.datasets as skd

import openml
from openml import tasks, runs

# importing libraries for convex optimization
import cvxpy

# importing deep learning libraries
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split


class Load_Dataset(Dataset):
    '''
    The dataloader returns a pytorch dataset
    inputs:
        ds_id: [str/int] refers to the selected dataset
            Synthetic datasets
            - 'hypercube'
            - 'blobs'
            - 'blobs2'
            Sklearn's datasets
            - 'iris'
            - 'digits'
            - 'covtype'
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

    def __init__(self, ds_id, shuffle=True, seed=None):
        self.ds_id = ds_id
        self.le = None
        self.seed = seed
        self.shuffle = shuffle
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
        X = None
        y = None
        if isinstance(ds_id, int):
            data = openml.datasets.get_dataset(ds_id)
            X, y, categorical, feature_names = data.get_data(
                target=data.default_target_attribute,
            )
            X = X.values
            # encode target labels as values
            self.le = preprocessing.LabelEncoder()
            y = self.le.fit_transform(y)
        elif ds_id == 'hypercube':
            X, y = skd.make_classification(
                n_samples=400, n_features=40, n_informative=40,
                n_redundant=0, n_repeated=0, n_classes=4,
                n_clusters_per_class=2,
                weights=None, flip_y=0.0001, class_sep=1.0, hypercube=True,
                shift=0.0, scale=1.0, shuffle=True, random_state=None)
        elif ds_id == 'blobs':
            X, y = skd.make_blobs(
                n_samples=400, n_features=2, centers=20, cluster_std=2,
                center_box=(-10.0, 10.0), shuffle=True, random_state=None)
        elif ds_id == 'blobs2':
            X, y = skd.make_blobs(
                n_samples=400, n_features=4, centers=10, cluster_std=1,
                center_box=(-10.0, 10.0), shuffle=True, random_state=None)
        elif ds_id == 'iris':
            dataset = skd.load_iris()
            X = dataset.data
            y = dataset.target
        elif ds_id == 'digits':
            dataset = skd.load_digits()
            X = dataset.data
            y = dataset.target
        elif ds_id == 'covtype':
            dataset = skd.fetch_covtype()
            X = dataset.data
            y = dataset.target - 1
        elif ds_id == 'mnist':
            train_dataset = torchvision.datasets.MNIST('/files/', train=True, download=True,
                                                       transform=torchvision.transforms.Compose([
                                                           torchvision.transforms.ToTensor(),
                                                           torchvision.transforms.Normalize(
                                                               (0.1307,), (0.3081,))]))
            # [TBD]
            X = torch.Tensor([np.array(a[0]) for a in train_dataset]).view(len(train_dataset), -1)
            y = torch.Tensor([a[1] for a in train_dataset])
        else:
            print('TBD. Sorry for the inconvenience.')

        self.n_samples = X.shape[0]
        self.n_classes = len(np.unique(y))

        if self.shuffle:
            X, y = sklearn.utils.shuffle(X, y, random_state=self.seed)
        self.X = X
        self.y = y

    def __getitem__(self, index):
        if self.z is None:
            return self.X[index], self.y[index]
        else:
            return self.X[index], self.z[index], self.y[index]

    def __len__(self):
        return self.n_samples

    def __shape__(self):
        return [self.n_samples, self.n_samples]

    def inverse_labels(self, labels):
        if self.le is not None:
            return self.le.inverse_transform(labels)

    def include_weak(self, z):
        self.z = z

    def generate_loaders(self, train_size, batch_size=1, shuffle=False):
        if train_size < 1:
            tr_s = int(len(dataset)*0.5)
        te_s = self.n_samples - tr_s
        trainloader = DataLoader(dataset=dataset[:tr_s], batch_size=batch_size, shuffle=shuffle)
        testloader = DataLoader(dataset = dataset[tr_s:], batch_size = batch_size, shuffle = shuffle)
        train_ds = torch.utils.data.random_split(dataset, [30, 120], generator=torch.Generator().manual_seed(42))
        test_ds = torch.utils.data.random_split(dataset, [30, 120], generator=torch.Generator().manual_seed(42))
        return train_ds, test_ds, trainloader, testloader