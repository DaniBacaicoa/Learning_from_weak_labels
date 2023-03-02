"""" Transforms a dataset with the true labels into a weakly labeled dataset
    The weakening process is given by a Mixing matrix, a.k.a., Transition matrix
    Authors: Daniel Bacaicoa-Barber, June 2022
             Jesús Cid-Sueiro (Original code)
             Miquel Perelló-Nieto (Original code)
"""
import numpy as np
import torch
import cvxpy
from utils.utils_weakener import label_matrix, pll_weights, binarize_labels


class Weakener(object):
    '''
    The weakener class serves for getting a weakly labeled dataset from another
    with the actual labels.
    '''

    def __init__(self, true_classes):
        '''
        Types of corruptions supported are:
        Given by a mixing matrix
            - Supervised
            Noise (d=c)
            - PU (c=2)
            - Complementary
            - Noisy
            Weak
            - Weak
            - PLL (with and without anchor points)
            - Complementary_weak
        Dependent on the sample [TBD]
            - See papers of this (how this works), maybe include some relaxations in M

        '''
        # This is for a matrix given by the user.
        self.c = true_classes
        self.M = None

        self.z = None
        self.w = None

    def generate_M(self,  model_class='supervised', alpha=1, beta=None):
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
            #self.M, self.Z, self.labels = label_matrix(M)
            #self.M = M
        # c = d
        elif model_class == 'supervised':
            M = np.identity(self.c)
            #self.M, self.Z, self.labels = label_matrix(M)

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
            #self.M, self.Z, self.labels = label_matrix(M)
            #self.M = M

        elif model_class == 'complementary':
            '''
            This gives one of de non correct label 
            '''
            M = (1 - np.eye(c)) / (c - 1)
            #self.M, self.Z, self.labels = label_matrix(M)
            #self.M = M

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
            #self.M, self.Z, self.labels = label_matrix(M)
            #self.M = M


        elif model_class == 'pll':
            # Mixing matrix for making pll corruption similar to that in
            # Instance-Dependent PLL (Xu, et al. 2021)
            probs, Z = pll_weights(self.c, p=0.5)  # Take this probability from argparse

            M = np.array([list(map(probs.get, Z[:, i] * np.sum(Z, 1))) for i in range(self.c)]).T
            #self.M, self.Z, self.labels = label_matrix(M)

        elif model_class == 'pll_a':
            # Mixing matrix for making pll corruption similar to that in
            # Instance-Dependent PLL (Xu, et al. 2021) they don't allow anchor points but this method does.
            probs, Z = pll_weights(self.c, p=0.5, anchor_points=True)  # Take this probability from argparse

            M = np.array([list(map(probs.get, Z[:, i] * np.sum(Z, 1))) for i in range(self.c)]).T
            #self.M, self.Z, self.labels = label_matrix(M)

        elif model_class == 'Complementary_weak':
            '''
            This gives a set of candidate labels over the non correct one.
            '''
            # [TBD]
            return _
        self.M, self.Z, self.labels = label_matrix(M)

    def generate_weak(self, y, seed=None):
        # It should work with torch
        # the version of np.random.choice changed in 1.7.0 that could raise an error-
        d, c = self.M.shape
        # [TBD] include seed
        self.z = torch.Tensor([np.random.choice(d, p=self.M[:, tl]) for tl in torch.max(y, axis=1)[1]]).to(torch.int32)

        self.w = torch.from_numpy(self.Z[self.z.to(torch.int32)] + 0.)
        return self.z, self.w


    def virtual_matrix(self, p=None, convex=True):
        d, c = self.M.shape
        I_c = np.eye(c)
        if p == None:
            p = np.ones(d)/d
        c_1 = np.ones((c,1))
        d_1 = np.ones((d,1))

        hat_Y = cvxpy.Variable((c,d))

        if c==d:
            self.Y = np.linalg.pinv(self.M)
        elif convex:
            prob = cvxpy.Problem(cvxpy.Minimize(
                cvxpy.norm(cvxpy.hstack([cvxpy.norm(hat_Y[:, i])**2 * p[i] for i in range(d)]),1)
            ),
                [hat_Y @ self.M == I_c, hat_Y.T @ c_1 == d_1]
            )
            prob.solve()
            self.Y = hat_Y.value
        else:
            prob = cvxpy.Problem(cvxpy.Minimize(
                cvxpy.norm(cvxpy.hstack([cvxpy.norm(hat_Y[:, i])**2 * p[i] for i in range(d)]),1)
            ),
                [hat_Y @ self.M == I_c]
            )
            prob.solve()
            self.Y = hat_Y.value


    def virtual_labels(self, y = None):
        '''
        z must be the weak label in the z form given by generate weak
        '''
        #In order to not generate weak labels each time we seek the existence of them
        # and in the case they are already generated we dont generate them again
        if self.z is None:
            if y is None:
                raise NameError('The weak labels have not been yet created. You shuold give the true labels. Try:\n  class.virtual_labels(y)\n instead')
            _,_ = self.generate_weak(y)
        self.virtual_matrix()
        self.v = self.Y.T[self.z]
        return

    def generate_wl_priors(self,z):
        if self.z is None:
            _,_ = self.generate_weak(y, seed=seed)


