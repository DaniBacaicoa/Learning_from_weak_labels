import torch
import torch.nn as nn
import numpy as np

import datasets.datasets as dtset
import utils.losses as losses

from utils.weakener import Weakener
from models.model import MLP

from utils.trainig_testing import train_and_evaluate

import pickle
f = open("Experimental_results/Datasets.pkl","rb")
Data,Weak = pickle.load(f)
f.close()

Weak.virtual_labels(p=None, optimize = True, convex = True) #This is to create Virtual Labels
#Data.include_weak(Weak.z)
Data.include_virtual(Weak.v)
trainloader,testloader = Data.get_dataloader(weak_labels='virtual')

loss = losses.LBLoss(k=1,beta=1.2)
overall_results = {}
overall_models = {}
epochs = 50
for i in range(10):
    mlp = MLP(Data.num_features,[Data.num_features],Data.num_classes, dropout_p=0.5, bn=True, activation =  'gelu')
    optim = torch.optim.Adam(mlp.parameters(),lr=1e-2)
    mlp, results = train_and_evaluate(mlp,trainloader,testloader,optimizer=optim,loss_fn=loss,num_epochs=epochs,sound=1)
    overall_results[i] = results
    overall_models[i] = mlp


LBL = [overall_results,overall_models]
f = open("Experimental_results/LBL.pkl","wb")
pickle.dump(LBL,f)
f.close()