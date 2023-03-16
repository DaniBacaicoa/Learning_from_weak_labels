import torch
from copy import deepcopy
from datasets.benchmark import Bk_Dataset
from datasets.openml_datasets import OpenML_Dataset
from utils.weakener import Weakener
from utils import losses
from utils.arguments import argument_parser

args = argument_parser()
#device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')










if __name__ == "__main__":
    #Data loading
    try:
        DS = Bk_Dataset(args.ds)
    except TypeError:
        DS = OpenML_Dataset(args.ds)

    #Optimizer and loss configuration
    optimizer_class = getattr(torch.optim, args.optim)
    optimizer = optimizer_class(**args.optim_params)

    loss_class = getattr(losses,args.loss)
    if args.loss == 'LBLoss':
        loss_fn = loss_class(args.lbl_params)
    elif args.loss == 'PartialLoss':
        loss_fn = loss_class(weak.w)


