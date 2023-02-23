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


def benchmark(config):
    DS = Bk_Dataset(dataset = config.id)
    train_X, train_y, test_X, test_y = DS.get_data()
    WK = Weakener(DS.num_classes)
    WK.generate_M(model_class = config.type)
    _, weak_y = WK.generate_weak(train_y)
    DS.include_weak(weak_y)
    trainloader, testloader = DS.get_dataloader()

    net = mlp_feature(DS.num_features,DS.num_features,DS.num_classes)
    enc = deepcopy(net)
    net,enc = map(lambda x: x.to(device),(net,enc))

    partialize_net = mlp_phi(DS.num_features,DS.num_classes)

    # Warm up
    net,feature_extracted,o_array = warm_up_benchmark(config,net,trainloader,testloader)

    # Now the training with our model

    
    


def weaklabels(config):
    DS = Load_Dataset(config.id)

    Wea









if __name__ == "__main__":
    if args.dt == "benchmark":
        benchmark(args)
    if args.dt == "open_ml":
        weaklabes(args)
