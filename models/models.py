import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, n_layers, dropout_p, dataset_name):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.dataset_name = dataset_name

        self.layers = nn.ModuleList([nn.Linear(input_size, hidden_sizes[0])])
        self.layers.extend([nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]) for i in range(n_layers - 2)])
        self.layers.extend([nn.Linear(hidden_sizes[-1], output_size)])
        self.dropout = nn.Dropout(p=dropout_p)
        self.batch_norm = nn.BatchNorm1d(hidden_sizes[0])

        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        for i, layer in enumerate(self.layers[:-1]):
            x = F.relu(layer(x))
            x = self.dropout(x)
            x = self.batch_norm(x)
        x = self.layers[-1](x)
        return x

    def train_epoch(self, trainloader, optimizer):
        loss_epoch = 0
        for data, target in trainloader:
            output = self(data)
            loss = loss(data, target)
            loss_epoch += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return loss_epoch

    def eval_epoch(self, testloader):
        loss_epoch = 0
        correct = 0
        with torch.no_grad():
            for data, target in testloader:
                loss, correct_batch = self.eval_step(data, target)
                loss_epoch += loss.item()
                correct += correct_batch
        return loss_epoch, correct

    def eval_step(self, data, target):
        self.eval()
        output = self(data)
        loss = F.cross_entropy(output, target)
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        return loss, correct


class benchmark_MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, n_layers, dropout_p, dataset_name):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.dataset_name = dataset_name

        self.layers = nn.ModuleList([nn.Linear(input_size, hidden_sizes[0])])
        self.layers.extend([nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]) for i in range(n_layers - 2)])
        self.layers.extend([nn.Linear(hidden_sizes[-1], output_size)])
        self.dropout = nn.Dropout(p=dropout_p)
        self.batch_norm = nn.BatchNorm1d(hidden_sizes[0])

        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        for i, layer in enumerate(self.layers[:-1]):
            x = F.relu(layer(x))
            x = self.dropout(x)
            x = self.batch_norm(x)
        x = self.layers[-1](x)
        return x

    def train_epoch(self, trainloader, optimizer):
        loss_epoch = 0
        for data, target in trainloader:
            output = self(data)
            loss = loss(data, target)
            loss_epoch += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return loss_epoch

    def eval_epoch(self, testloader):
        loss_epoch = 0
        correct = 0
        with torch.no_grad():
            for data, target in testloader:
                loss, correct_batch = self.eval_step(data, target)
                loss_epoch += loss.item()
                correct += correct_batch
        return loss_epoch, correct

    def eval_step(self, data, target):
        self.eval()
        output = self(data)
        loss = F.cross_entropy(output, target)
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        return loss, correct

