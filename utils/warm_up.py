import torch
import torch.nn.functional as F
import torch.nn as nn
from utils.utils_weakener import binarize_labels
from utils.args import parse_args
from utils.losses import PartialLoss

#args = parse_args()
#device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')


def warm_up_benchmark(model, train_loader, testloader):
    opt = torch.optim.SGD(list(model.parameters()), lr=1e-2, weight_decay=1e-4, momentum=0.9)
    #optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001)
    loss_fn = PartialLoss(train_loader.dataset[:][1])
    #partial_weight = train_loader.dataset[:][1].clone().detach()
    #partial_weight = partial_weight / partial_weight.sum(dim=1, keepdim=True)
    print("Begin warm-up, warm up epoch {}".format(10))

    for _ in range(0, 10):
        for features, targets, trues, indexes in train_loader:
            opt.zero_grad()
            outputs = model(features)

            L_ce = loss_fn(outputs, targets, indexes)

            L_ce.backward()
            opt.step()

        for features, targets in testloader:
            with torch.no_grad():
                outputs = model(features)
                _, pred = torch.max(F.softmax(outputs, dim=1),dim=1)


                pred = binarize_labels(10,pred.to(torch.long))

                test_acc = (pred*targets).sum()/len(testloader.dataset)
        print(targets[0:3, :])
        print(pred[0:3, :])
        print("After warm up, test acc: {:.4f}".format(test_acc))

    return model
