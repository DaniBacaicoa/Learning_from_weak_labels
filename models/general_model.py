import torch.nn as nn
#import torch.nn.init as init
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_p=0.0,
                 activation = 'relu', layer_init=nn.init.xavier_uniform_, bn_init=init.ones_,
                 bn_momentum = 0.1, seed = None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)

        # Create a list of layer sizes
        if type(hidden_sizes) == list:
            layer_sizes = [input_size] + hidden_sizes + [output_size]
        else:
            layer_sizes = [input_size] + [hidden_sizes] + [output_size]

        # Create a list of linear layers using ModuleList
        self.layers = nn.ModuleList([
            nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            for i in range(len(layer_sizes) - 1)
        ])
        # Initialize layers using given initialization function
        for layer in self.layers:
            layer_init(layer.weight)

        # Create a list of batch normalization layers using ModuleList
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(layer_sizes[i + 1], momentum=bn_momentum)
            for i in range(len(hidden_sizes))
        ])

        # Initialize batch normalization layers using given initialization function
        for bn in self.batch_norms:
            bn_init(bn.weight)

        # Create a dropout layer
        self.dropout = nn.Dropout(dropout_p)


        self.activation = activation
        self.bn = bn

    def forward(self, x):
        # Iterate over the linear layers and apply them sequentially to the input
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            if self.bn:
                x = self.batch_norms[i](x)
            activation_fn = getattr(nn.functional, self.activation)
            x = activation_fn(x)
            x = self.dropout(x)
        # Apply the final linear layer to get the output
        x = self.layers[-1](x)
        return x


# We can instantiate to have mlp_phi
# mlp_phi = MLP(n_inputs=4, n_outputs=2, layer_sizes=[300, 301, 302, 303], dropout_p=0.0, activation_fn=F.relu,
#              layer_init=init.xavier_uniform_, bn_init=init.ones_, bn_momentum=0.1)
#
#mlp_feature = MLP(input_dim, [hidden_dim, hidden_dim, hidden_dim], output_dim, activations=[nn.ReLU(), nn.ReLU(), nn.ReLU()])