import os
import torch
import torch.nn as nn
import numpy as np
import argparse

import datasets.datasets as dtset
import utils.losses as losses
from utils.weakener import Weakener
from models.model import MLP
from utils.trainig_testing import train_and_evaluate
from Dataset_generation import generate_dataset
import pickle
import json

def main(reps, epochs, dropout_p, loss_type, pll_p, k=1, beta=1.2, lr= 5e-2, betas = (0.8, 0.99)):

    json_results_dict = {
        'arguments': {
            'reps': reps,
            'epochs': epochs,
            'dropout_p': dropout_p,
            'loss_type': loss_type,
            'pll_p': pll_p,
            'k': k,
            'beta': beta,
            'lr': lr,
            'betas': betas
        },
    }


    save_dir = f'Experimental_results({pll_p})'
    # Create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        #This is for creating an mnist dataset inside the folder if it does'nt exist
        generate_dataset(save_dir, pll_p = pll_p,) #This

    #Reading the mnist dataset so all losses work with the same data
    f = open(save_dir + "/Dataset.pkl","rb")
    Data,Weak = pickle.load(f)
    f.close()

    # Choose loss function 
    if loss_type in ['Back','Back_conv','Back_opt','Back_opt_conv']:
        loss_fn = losses.CELoss()
        if loss_type == 'Back':
            Weak.virtual_labels(p=None, optimize = False, convex = False)
        elif loss_type == 'Back_opt':
            Weak.virtual_labels(p=None, optimize = True, convex = False)
        elif loss_type == 'Back_conv':
            Weak.virtual_labels(p=None, optimize = False, convex = True)
        elif loss_type == 'Back_opt_conv':
            Weak.virtual_labels(p=None, optimize = True, convex = True)
        Data.include_virtual(Weak.v)
        trainloader,testloader = Data.get_dataloader(weak_labels='virtual')
    elif loss_type == 'EM':
        loss_fn = losses.EMLoss(Weak.M)
        Data.include_weak(Weak.z)
        trainloader,testloader = Data.get_dataloader(weak_labels='weak')
    elif loss_type == 'OSL':
        loss_fn = losses.OSLCELoss()
        Data.include_weak(Weak.z)
        trainloader,testloader = Data.get_dataloader(weak_labels='weak')
    elif loss_type == 'LBL':
        loss_fn = losses.LBLoss(k, beta)
        Weak.virtual_labels(p=None, optimize = True, convex = True)
        Data.include_virtual(Weak.v)
        trainloader,testloader = Data.get_dataloader(weak_labels='virtual')
    elif loss_type == 'Forward':
        loss_fn = losses.ForwardLoss(Weak.M)
        Data.include_weak(Weak.z)
        trainloader,testloader = Data.get_dataloader(weak_labels='weak')
    elif loss_type == 'ForwardBackward':
        #Weak.V_matrix(Data.num_classes)
        Y = np.linalg.pinv(Weak.M)
        loss_fn = losses.FBLoss(Weak.M, Y)
        Data.include_weak(Weak.z)
        trainloader,testloader = Data.get_dataloader(weak_labels='weak')
    else:
        raise ValueError("Invalid loss type. Check the spelling")

    overall_results = {}
    overall_models = {}

    for i in range(reps):
        np_results = {}
        mlp = MLP(Data.num_features, [Data.num_features], Data.num_classes, dropout_p = dropout_p, bn = True, activation = 'tanh')
        optim = torch.optim.Adam(mlp.parameters(), lr = lr, betas = betas)
        mlp, results = train_and_evaluate(mlp, trainloader, testloader, optimizer=optim, loss_fn=loss_fn, num_epochs=epochs, sound=1)
        print(results)
        for i in results:
            np_results[i] = results[i].numpy()

        overall_results[i] = np_results
        overall_models[i] = mlp

    # Save results
    results_dict = {'overall_results': overall_results, 'overall_models': overall_models}
    json_results_dict['results'] = overall_results
    save_path = os.path.join(save_dir, f'{loss_type}.pkl')
    json_save_path = os.path.join(save_dir, f'json_{loss_type}.json')
    with open(save_path, "wb") as f:
        pickle.dump(results_dict, f)
    #with open(json_save_path, "w") as f:
    #    json.dump(json_results_dict, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate MLP model')
    parser.add_argument('--reps', type = int, default = 5, help = 'Number of repetitions for statistical purposes')
    parser.add_argument('-e','--epochs', type = int, default = 15, help = 'Number of epochs')
    parser.add_argument('-dp', '--dropout', type = float, default = 0.5, help = 'Dropout probability')
    parser.add_argument('-l','--loss', type = str, default = 'Back', 
                        choices=['Back','Back_conv','Back_opt','Back_opt_conv','EM','OSL','LBL','Forward','ForwardBackward'],
                        help='Type of loss reconstruction')
    #parser.add_argument('-d', '--save_dir', type = str, default = 'Experimental_results', help = 'Directory to save results')
    parser.add_argument('--pll', type = float, help = 'Probability of corrupted samples in the dataset')
    parser.add_argument('--k', type = float, default = 1, help = 'Hyperparameter for LBL (lower bounded loss)')
    parser.add_argument('--beta', type = float, default = 1.2, help = 'Hyperparameter for LBL (lower bounded loss)')
    parser.add_argument('--lr', type = float, default = 1e-3, help = 'Learning rate for the optimization algorithm')
    parser.add_argument('--betas', type = tuple, default = (0.9, 0.999), help = 'betas parameter for Adam optimizer')
    args = parser.parse_args()

    main(args.reps, args.epochs, args.dropout, args.loss, args.pll, args.k, args.beta, args.lr, args.betas)


#To acces main's hepl:
# python main.py --help