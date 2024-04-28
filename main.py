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

def main(reps, epochs, dropout_p, loss_type, save_dir, pll_p, k=1, beta=1.2):
    # Create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        generate_dataset(save_dir, pll_p = pll_p,)

    f = open(save_dir + "/Dataset.pkl","rb")
    Data,Weak = pickle.load(f)
    f.close()

    # Choose loss function
    if loss_type in ['Back','Back_conv','Back_opt','Back_opt_conv']:
        loss_fn = losses.CELoss()
    elif loss_type == 'EM':
        loss_fn = losses.EMLoss(Weak.M)
    elif loss_type == 'OSL':
        loss_fn = losses.OSLCELoss()
    elif loss_type == 'LBL':
        loss_fn = losses.LBLoss(k, beta)
    elif loss_type == 'Forward':
        loss_fn = losses.ForwardLoss(V)
    elif loss_type == 'ForwardBackward':
        loss_fn = losses.FBLoss(Weak.M, V)
    else:
        raise ValueError("Invalid loss type. Check the spelling")

    overall_results = {}
    overall_models = {}

    for i in range(reps):
        mlp = MLP(Data.num_features, [Data.num_features], Data.num_classes, dropout_p=dropout_p, bn=True, activation='gelu')
        optim = torch.optim.Adam(mlp.parameters(), lr=1e-2)
        mlp, results = train_and_evaluate(mlp, trainloader, testloader, optimizer=optim, loss_fn=loss_fn, num_epochs=epochs, sound=1)
        overall_results[i] = results
        overall_models[i] = mlp

    # Save results
    results_dict = {'overall_results': overall_results, 'overall_models': overall_models}
    save_path = os.path.join(save_dir, "Back_conv.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(results_dict, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate MLP model')
    parser.add_argument('--reps', type=int, default=3, help='Number of repetitions')
    parser.add_argument('-ep','--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('-dp', '--dropout', type=float, default=0.5, help='Dropout probability')
    parser.add_argument('-l','--loss', type=str, default='CELoss', choices=['CELoss', 'YourCustomLoss'], help='Type of loss function (CELoss/YourCustomLoss)')
    parser.add_argument('-d', '--save_dir', type=str, default='Experimental_results(0.5)', help='Directory to save results')

    args = parser.parse_args()

    main(args.reps, args.epochs, args.dropout, args.loss, args.save_dir)


    python train_model.py --help