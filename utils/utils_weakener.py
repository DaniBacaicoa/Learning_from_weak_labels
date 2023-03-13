import torch
import numpy as np

def label_matrix(M):
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


def pll_weights(c, p=0.5, anchor_points=False):
    '''

    :param c:
    :param p:
    :param anchor_points: Whether presence of anchor points are allowed
    :return:
    '''
    _,Z,_ = label_matrix(np.ones((2 ** c, c)))
    probs = {0: 0}
    q = 1 - p
    for i in range(1, c + 1):
        if anchor_points:
            probs[1] = q ** c + p * q ** (c - 1)
            probs[2] = p ** 2 * q ** (c - 2) + p * q ** (c - 1)
        else:
            probs[1] = 0
            probs[2] = p ** 2 * q ** (c - 2) + p * q ** (c - 1) + (q ** c + p * q ** (c - 1)) / (c - 1)
        probs[i] = p ** i * q ** (c - i) + p ** (i - 1) * q ** (c - i + 1)
    return probs, np.array(Z)

def binarize_labels(c, y):
    A = torch.eye(c)[y]
    return A


