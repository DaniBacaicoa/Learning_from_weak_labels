"""" Transforms a dataset with the true labels into a weakly labeled dataset
    The weakening process is given by a Mixing matrix, a.k.a., Transition matrix
    Authors: Daniel Bacaicoa-Barber, June 2022
             Jesús Cid-Sueiro (Original code)
             Miquel Perelló-Nieto (Original code)
"""
import numpy as np
import torch
from utils.utils_weakener import label_matrix, pll_weights,binarize_labels


class Weakener(object):
    '''
    The weakener class serves for getting a weakly labeled dataset from another
    with the actual labels.
    '''

    def __init__(self, true_classes, M=None):
        '''
        Types of corruptions supported are:
        Given by a mixing matrix
            - Supervised
            Noise
            - PU
            - Complementary
            - Noisy
            Weak
            - IPL
            - PLL (with and without anchor points)
            -
        Dependent on the sample [TBD]
            - See papers of this (how this works), maybe include some relaxations in M

        '''
        # This is for a matrix given by the user.
        self.c = true_classes
        if M is not None:
            r, c = M.shape
            if c != self.c:
                raise NameError('The transition matrix given doesnt suit the number of true classes')
            else:
                self.M = M
        else:
            self.M = None

    def Load_M(self, M):
        '''
        User given transition matrix
        '''
        self.M = M

    def generate_M(self, alpha=1, beta=None, model_class='supervised'):
        '''
        Generates a corruption matrix (a transition matrix)

        Parameters
        ----------
        weak_dist = float or array type.
            It goberns the distribution of the
        alpha
        beta
        model_class
        '''
        # with fixed size c=2
        if model_class == 'pu':
            if self.c > 2:
                raise NameError('PU corruption coud only be applied when tne number o true classes is 2')
                # [TBD] if alpha is a vector raise error
                alpha = [alpha, 0]
            M = np.eye(2) + alpha * np.ones((2, 2))
            M /= np.sum(M, 0)
            self.M = M
        # c = d
        elif model_class == 'supervised':
            self.M = np.identity(self.c)
        elif model_class == 'noisy':
            '''
            - alpha > -1
                the limit case alpha = -1: Complemetary labels
                if alpha < 0: The false classes are more probable
                if alpha = 0: all classes are equally probable
                if alpha > 0: The true class is more prbable. 
                As a limiting case supervised is achieved as alpha -> infty
            [  1+a_1  a_2    a_3  ]
            [  a_1    1+a_2  a_3  ]
            [  a_1    a_2    1+a_3]
            '''
            if any(np.array(alpha) < -1):
                NameError('For noisy labels all components of alpha shoud be greater than -1')
            elif any(np.array(alpha) == -1):
                cl = np.where(np.array(alpha) == -1)[0]
                print('labels', cl, 'are considered complemetary labels')
                # warning('Some (or all) of the components is considered as complemetary labels')
            M = np.eye(self.c) + alpha * np.ones(self.c)
            M /= np.sum(M, 0)
            self.M = M

        elif model_class == 'complementary':
            '''
            This gives one of de non correct label 
            '''
            self.M = (1 - np.eye(c)) / (c - 1)

        # c < d
        elif model_class == 'weak':
            '''
            - alpha > -1
                the limit case alpha = -1: Complemetary labels
                if alpha < 0: The false classes are more probable
                if alpha = 0: all classes are equally probable
                if alpha > 0: The true class is more probable. 
                As a limiting case supervised is achieved as alpha -> infty

             z\y  001    010    100
            000[  a_1    a_2    a_3  ]
            001[  1+a_1  a_2    a_3  ]
            010[  a_1    1+a_2  a_3  ]
            001[  a_1    a_2    a_3  ]
            011[  a_1    a_2    a_3  ]
            100[  a_1    a_2    1+a_3]
            101[  a_1    a_2    a_3  ]
            111[  a_1    a_2    a_3  ]
            '''
            M = np.zeros((2 ** self.c, self.c))
            for i in range(self.c):
                M[2 ** i, i] = 1
            M = alpha * M + np.ones((2 ** self.c, self.c))
            M /= np.sum(M, 0)
            self.M = M


        elif model_class == 'pll':
            # Mixing matrix for making pll corruption similar to that in
            # Instance-Dependent PLL (Xu, et al. 2021)
            probs, Z = pll_weights(self.c, p=0.5)  # Take this probability from argparse

            M = np.array([list(map(probs.get, Z[:, i] * np.sum(Z, 1))) for i in range(self.c)]).T
            self.M, self.Z, self.labels = label_matrix(M)

        elif model_class == 'pll_a':
            # Mixing matrix for making pll corruption similar to that in
            # Instance-Dependent PLL (Xu, et al. 2021) they don't allow anchor points but this method does.
            probs, Z = pll_weights(self.c, p=0.5, anchor_points=True)  # Take this probability from argparse

            M = np.array([list(map(probs.get, Z[:, i] * np.sum(Z, 1))) for i in range(self.c)]).T
            print(M)
            self.M, self.Z, self.labels = label_matrix(M)

        elif model_class == 'Complementary_weak':
            '''
            This gives a set of candidate labels over the non correct one.
            '''
            # [TBD]
            return _

    def generate_weak(self, y, seed=None):
        # It should work with torch
        # the version of np.random.choice changed in 1.7.0 that could raise an error-
        d, c = self.M.shape
        # [TBD] include seed
        p_Y = torch.Tensor([np.random.choice(d, p=self.M[:, tl]) for tl in torch.max(y, axis=1)[1]])

        p_Y_oh = torch.from_numpy(self.Z[p_Y.to(torch.int32)] + 0.)
        return p_Y, p_Y_oh

    #         d, c = self.M.shape
    #         wl = np.arange(self.c)
    #         # [TBD] include seed
    #         return np.array([np.random.choice(wl, p=self.M[:, tl]) for tl in y])

    def binarize_labels(self, ):  # I have another binarize in another place
        return ()

    def weak_dataset(self, X=None, y=None, z=None, Dataset=None, batch_size=None, seed=None):
        '''
        Receives either a pytorch dataset with the true classes or a feature matrix, X and a label vector y
        And returns a Dataset with three instances in order to make the training and the test
        '''
        if z is None:
            z = weak.generate_weak(y, seed=None)
        if Dataset is None:
            weak_dataset = torch.utils.data.TensorDataset(torch.tensor(X), torch.tensor(z), torch.tensor(y))
        weak_loader = torch.utils.data.DataLoader(weak_dataset, batch_size=batch_size, shuffle=True)

        if batch_size is None:
            return weak_dataset
        else:
            return weak_loader

    # def train_test(self, dataset, train_prop = 0.7):
    #    train_set, val_set = torch.utils.data.random_s plit(dataset, [50000, 10000])

    def virtual_matrix(self, p=None, convex=True):
        d, c = self.M.shape
        if p is None:
            p = np.ones(d)
            p /= np.sum(p)
        d1 = np.ones(d)

        if d == c:
            # we could use inv, when generated by Generate_M we'll never have a singular matrix
            # but when given by user it could be.
            Y = np.linalg.pinv(self.M)
        else:
            # [TBD] Include the different reconstructions
            Y = np.linalg.pinv(self.M)
