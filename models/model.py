import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_p=0.0, bn = False):
        super().__init__()

        # Create a list of layer sizes
        layer_sizes = [input_size] + hidden_sizes + [output_size]

        # Create a list of linear layers using ModuleList
        self.layers = nn.ModuleList([
            nn.Linear(layer_sizes[i], layer_sizes[i+1])
            for i in range(len(layer_sizes)-1)
        ])
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)

        # Create a list of batch normalization layers using ModuleList
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(layer_sizes[i+1])
            for i in range(len(hidden_sizes))
        ])

        # Create a dropout layer
        self.dropout = nn.Dropout(dropout_p)
        self.bn = bn

    def forward(self, x):
        # Iterate over the linear layers and apply them sequentially to the input
        for i in range(len(self.layers)-1):
            x = self.layers[i](x)
            if self.bn:
                x = self.batch_norms[i](x)
            x = nn.functional.relu(x)
            x = self.dropout(x)
        # Apply the final linear layer to get the output
        x = self.layers[-1](x)
        return x

class mlp_valen(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(mlp_valen, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        #out = x.view(x.size(0), -1)
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        #In Valen this gives features and output 
        # but I cannot see it clearly
        out = self.fc4(out)
        return  out
