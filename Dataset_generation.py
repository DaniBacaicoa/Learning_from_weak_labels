import torch
import torch.nn as nn
import numpy as np

import datasets.datasets as dtset
import utils.losses as losses

from utils.weakener import Weakener
from models.model import MLP

Data = dtset.Torch_Dataset('mnist', batch_size = 16)
Weak = Weakener(Data.num_classes)
Weak.generate_M(model_class='pll',pll_p=0.5)
train_X,train_y,test_X,test_y =  Data.get_data()
Weak.generate_weak(train_y) #z and w


import pickle
Dataset = [Data,Weak]
f = open("Experimental_results(0.5)/Datasets.pkl","wb")
pickle.dump(Dataset,f)
f.close()