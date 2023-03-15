import torch
from copy import deepcopy
from datasets.benchmark import Bk_Dataset
from datasets.openml_datasets import Load_Dataset
from utils.weakener import Weakener
from utils.warm_up import warm_up_benchmark
from models.benchmark_mlp import mlp_feature,mlp_phi
from utils.args import parse_args

args = parse_args()
device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')











if __name__ == "__main__":
    if args.dt == "benchmark":
        benchmark(args)
    if args.dt == "open_ml":
        weaklabes(args)
