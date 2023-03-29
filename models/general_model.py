import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_p=0.0, bn = True,
                 activation = 'relu', layer_init=nn.init.xavier_uniform_, bn_init=nn.init.ones_,
                 bn_momentum = 0.1, seed = None):
        super().__init__()
        self.bn = bn
        if seed is not None:
            torch.manual_seed(seed)

        # Create a list of layer sizes
        if type(hidden_sizes) == list:
            layer_sizes = [input_size] + hidden_sizes + [output_size]
        else:
            layer_sizes = [input_size] + [hidden_sizes] + [output_size]

        # We create a list of linear layers and initialize their weights
        self.layers = nn.ModuleList([
            nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            for i in range(len(layer_sizes) - 1)
        ])
        for layer in self.layers:
            layer_init(layer.weight)

        # We create a list of batch normalization layers and initialize their weights
        if self.bn:
            self.batch_norms = nn.ModuleList([
                nn.BatchNorm1d(layer_sizes[i + 1], momentum = bn_momentum)
                for i in range(len(hidden_sizes))
            ])
            for bn in self.batch_norms:
                bn_init(bn.weight)

        # Create a dropout layer
        self.dropout = nn.Dropout(dropout_p)

        self.activation = activation
        self.bn = bn

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            if self.bn:
                x = self.batch_norms[i](x)
            activation_fn = getattr(nn.functional, self.activation)
            x = activation_fn(x)
            x = self.dropout(x)
        x = self.layers[-1](x)
        return x

## For using Valen's mlp: (the seed is set for reproducibility reasons even if the authors didn't use it)
## Benchmark models [nmist,knmist,fnmist]
##    Partialization =  rand (instance independent), they use mlp_feature that can be generated instantiating MLP as:
##      mlp_feature = MLP(input_dim, [hidden_dim, hidden_dim, hidden_dim], output_dim, dropout_p = 0.0, bn = False, seed = 1,
##             layer_init = lambda x: nn.init.kaiming_uniform_(x, a=math.sqrt(5))
## Benchmark models [nmist,knmist,fnmist]
##    Partialization =  feature (instance dependent), they use mlp_phi that can be generated instantiating MLP as:
##       mlp_phi = MLP(input_dim, hidden_sizes=[300, 301, 302, 303],  output_dim, dropout_p = 0.0, seed = 1)
#
#