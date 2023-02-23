import torch
import torch.nn.functional as F
import torch.nn as nn
from utils.utils_weakener import binarize_labels
from utils.args import parse_args
from utils.losses import partial_loss

args = parse_args()
device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')


def warm_up_benchmark(config, model, train_loader, testloader):
    opt = torch.optim.SGD(list(model.parameters()), lr=config.lr, weight_decay=config.wd, momentum=0.9)
    partial_weight = train_loader.dataset[:][1].clone().detach().to(device)
    partial_weight = partial_weight / partial_weight.sum(dim=1, keepdim=True)
    print("Begin warm-up, warm up epoch {}".format(config.warm_up))
    for _ in range(0, config.warm_up):
        for features, targets, trues, indexes in train_loader:
            features, targets, trues = map(lambda x: x.to(device), (features, targets, trues))
            phi, outputs = model(features)
            L_ce, new_labels = partial_loss(outputs, partial_weight[indexes,:].clone().detach(), None)
            partial_weight[indexes,:] = new_labels.clone().detach()
            opt.zero_grad()
            L_ce.backward()
            opt.step()
    for features,targets in testloader:
        with torch.no_grad():
            features, targets = map(lambda x: x.to(device), (features, targets))
            _, outputs = model(features)
            pred = torch.max(F.softmax(outputs, dim=1),dim=1)
            pred = binarize_labels(pred)
            test_acc = (pred*targets).sum()/Y.size(0)
    print("After warm up, test acc: {:.4f}".format(test_acc))

    print("Extract feature.")
    feature_extracted = torch.zeros((train_loader.dataset.train_X.shape[0], phi.shape[-1])).to(device)
    with torch.no_grad():
        for features, targets, trues, indexes in train_loader:
            features, targets, trues = map(lambda x: x.to(device), (features, targets, trues))
            feature_extracted[indexes, :] = model(features)[0]
    return model, feature_extracted, part
