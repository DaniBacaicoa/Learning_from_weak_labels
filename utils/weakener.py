"""" Transforms a dataset with the true labels into a weakly labeled dataset
    The weakening process is given by a Mixing matrix, a.k.a., Transition matrix
    Authors: Daniel Bacaicoa-Barber, June 2022
             Jesús Cid-Sueiro (Original code)
             Miquel Perelló-Nieto (Original code)
"""
import numpy as np
import torch
import cvxpy

from collections import Counter


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
        # We will allocate variables for important data 
         # This is for a matrix given by the user.
        self.c = true_classes
        self.d = None
        self.M = None
         # This is for the labels (when possible we will store all the possibilities)
        self.z = None
        self.w = None

    def generate_M(self,  model_class='supervised', alpha=1, beta=None,pll_p = 0.5):
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
        self.pll_p = pll_p
        # with fixed size c=2
        if model_class == 'pu':
            if self.c > 2:
                raise NameError('PU corruption coud only be applied when tne number o true classes is 2')
                # [TBD] if alpha is a vector raise error
                alpha = [alpha, 0]
            M = np.eye(2) + alpha * np.ones((2, 2))
            M /= np.sum(M, 0)
            #self.M, self.Z, self.labels = self.label_matrix(M)
            #self.M = M
        # c = d
        elif model_class == 'supervised':
            M = np.identity(self.c)
            #self.M, self.Z, self.labels = self.label_matrix(M)

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
                NameError('For noisy labels all components of alpha should be greater than -1')
            elif any(np.array(alpha) == -1):
                cl = np.where(np.array(alpha) == -1)[0]
                print('labels', cl, 'are considered complemetary labels')
                # warning('Some (or all) of the components is considered as complemetary labels')
            M = np.eye(self.c) + alpha * np.ones(self.c)
            M /= np.sum(M, 0)
            #self.M, self.Z, self.labels = self.label_matrix(M)
            #self.M = M

        elif model_class == 'complementary':
            '''
            This gives one of de non correct label 
            '''
            M = (1 - np.eye(c)) / (c - 1)
            #self.M, self.Z, self.labels = self.label_matrix(M)
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
            #self.M, self.Z, self.labels = self.label_matrix(M)
            #self.M = M


        elif model_class == 'pll':
            # Mixing matrix for making pll corruption similar to that in
            # Instance-Dependent PLL (Xu, et al. 2021)
            probs, Z = self.pll_weights(p=self.pll_p)  # Take this probability from argparse

            M = np.array([list(map(probs.get, Z[:, i] * np.sum(Z, 1))) for i in range(self.c)]).T
            #self.M, self.Z, self.labels = self.label_matrix(M)

        elif model_class == 'pll_a':
            # Mixing matrix for making pll corruption similar to that in
            # Instance-Dependent PLL (Xu, et al. 2021) they don't allow anchor points but this method does.
            probs, Z = self.pll_weights(p=self.pll_p, anchor_points=True)  # Take this probability from argparse

            M = np.array([list(map(probs.get, Z[:, i] * np.sum(Z, 1))) for i in range(self.c)]).T
            #self.M, self.Z, self.labels = self.label_matrix(M)

        elif model_class == 'Complementary_weak':
            '''
            This gives a set of candidate labels over the non correct one.
            '''
            # [TBD]
            return _

        self.M, self.Z, self.labels = self.label_matrix(M)
        self.d = self.M.shape[0]

    def generate_weak(self, y, seed=None):
        # It should work with torch
        # the version of np.random.choice changed in 1.7.0 that could raise an error-
        d, c = self.M.shape
        # [TBD] include seed
        self.z = torch.Tensor([np.random.choice(d, p=self.M[:, tl]) for tl in torch.max(y, axis=1)[1]]).to(torch.int32)

        self.w = torch.from_numpy(self.Z[self.z.to(torch.int32)] + 0.)
        return self.z, self.w


    def virtual_matrix(self, p=None, optimize = True, convex=True):
        d, c = self.M.shape
        I_c = np.eye(c)

        if p == None:
            if optimize:
                p = self.generate_wl_priors(self.z)
            else:
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
        # and in the case they are already generated we don't generate them again
        if self.z is None:
            if y is None:
                raise NameError('The weak labels have not been yet created. You shuold give the true labels. Try:\n  class.virtual_labels(y)\n instead')
            _,_ = self.generate_weak(y)
        self.virtual_matrix()
        self.v = self.Y.T[self.z]
        return

    def generate_wl_priors(self, loss = 'CELoss'):

        #z_count = Counter(z)
        #p_est = np.array([z_count[x] for x in range(self.d)])
        p_est = np.array(torch.bincount(self.z))
        v_eta = cvxpy.Variable(int(self.c))
        if loss == 'CELoss':
            lossf = -p_est @ cvxpy.log(self.M @ v_eta)
        else:
            p_est = p_est / np.sum(p_est)
            lossf = cvxpy.sum_squares(p_est - self.M @ v_eta)

        problem = cvxpy.Problem(cvxpy.Minimize(lossf),
                                [v_eta >= 0, np.ones(self.c) @ v_eta == 1])
        problem.solve()

        # Compute the wl prior estimate
        p_reg = self.M @ v_eta.value

        return p_reg
    '''
        v_eta = cvxpy.Variable(self.c)
        if loss == 'cross_entropy':
            lossf = -p_est @ cvxpy.log(self.M @ v_eta)
        elif loss == 'square_error':
            p_est = p_est / np.sum(p_est)
            lossf = cvxpy.sum_squares(p_est - self.M @ v_eta)
        '''

    def label_matrix(self, M):
        '''

        :param M:
        :param give_M:
        :return:
        '''

        Z = []
        trimmed_M = []

        d, c = M.shape
        e = 0
        if d == c:
            labels = dict((a, ''.join([[str(ele) for ele in sub] for sub
                                       in np.eye(c, dtype=int).tolist()][a])) for a in range(c))
            trimmed_M = M
            Z = M
        else:
            z_row = M.any(axis=1)
            labels = {}
            for i in range(2 ** c):
                if z_row[i]:
                    b = bin(i)[2:]
                    b = str(0) * (c - len(b)) + b  # this makes them same length
                    labels[e] = b
                    e += 1
                    Z.append(list(map(int, list(b))))
                    trimmed_M.append(M[i, :])
        return np.array(trimmed_M), np.array(Z), labels

    def pll_weights(self, p=0.5, anchor_points=False):
        '''

        :param self.c:
        :param p:
        :param anchor_points: Whether presence of anchor points are allowed
        :return:
        '''
        _, Z, _ = self.label_matrix(np.ones((2 ** self.c, self.c)))
        probs = {0: 0}
        q = 1 - p
        for i in range(1, self.c + 1):
            if anchor_points:
                probs[1] = q ** self.c + p * q ** (self.c - 1)
                probs[2] = p ** 2 * q ** (self.c - 2) + p * q ** (self.c - 1)
            else:
                probs[1] = 0
                probs[2] = p ** 2 * q ** (self.c - 2) + p * q ** (self.c - 1) + (q ** self.c + p * q ** (self.c - 1)) / (self.c - 1)
            probs[i] = p ** i * q ** (self.c - i) + p ** (i - 1) * q ** (self.c - i + 1)
        return probs, np.array(Z)