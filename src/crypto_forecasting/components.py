import numpy as np
from scipy.linalg import sqrtm

import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """
    Helper class to reuse saving/loading model code
    """
    def __init__(self) -> None:
        super().__init__()

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, map_location=None):
        if map_location is not None:
            if map_location == 'cpu':
                self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
            else:
                self.load_state_dict(torch.load(path, map_location=map_location))
        else:
            self.load_state_dict(torch.load(path))


class LSTM(BaseModel):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=False, predict=False) -> None:
        """
        Wrapper for PyTorch implementation of LSTM to take advantage of save/load of BaseModel
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=batch_first)
        self.predict = predict
        if predict:
            self.fc = nn.Linear(10*hidden_size, 14) # hard coded to sequences of length 10

    def initialize_hidden_state(self, batch_size):
        self.batch_size = batch_size
        self.hidden_state = (torch.zeros(self.num_layers, batch_size, self.hidden_size), torch.zeros(self.num_layers, batch_size, self.hidden_size))

    def forward(self, x, hidden_state=None):
    """
    Run the LSTM over a sequence of flattened asset features.

    Args:
        x: torch.Tensor of shape (batch_size, seq_len, input_size), where:
            - batch_size is the number of sequences in the batch
            - seq_len is the number of time steps
            - input_size is the total feature dimension across all assets

    Returns:
        If predict=True:
            tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]
            - predictions of shape (batch_size, 14)
            - hidden state tuple (h_n, c_n), each of shape (1, batch_size, hidden_size)

        Otherwise:
            tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]
            - sequence output of shape (batch_size, seq_len, hidden_size)
            - hidden state tuple (h_n, c_n)
    """

        if hidden_state is not None:
            self.hidden_state = hidden_state
        output, hidden_state = self.lstm(x, self.hidden_state)
        self.hidden_state = (hidden_state[0].detach(), hidden_state[1].detach())
        if self.predict:
            return self.fc( output.view(self.batch_size, -1) ), self.hidden_state
        else:
            return output[:, -1, :], self.hidden_state # just output of last in sequence


class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, activation, adj=None) -> None:
        """
        Single layer of graph convolution that takes in node features and adjacency matrix and outputs some value

        Args:
            in_dim: int, number of features for node vector
            out_dim: int, number of values to predict for this node
            activation: str, activation function to use
            adj: np.ndarray, optional adjacency matrix if fixed for every iteration
        """

        super().__init__()

        if adj is not None:
            # constant adjacency convolutions
            adj_tensor = torch.as_tensor(adj, dtype = torch.float32)

            if adj_tensor[0, 0] == 0:
                adj_tensor = adj_tensor + torch.eye(adj_tensor.shape[0], dtype = adj_tensor.dtype)
            self.register_buffer("a", adj_tensor)
        
        else:
            self.a = None

        self.weight = nn.Parameter(torch.empty(in_dim, out_dim, dtype=torch.float))
        self.bias = nn.Parameter(torch.empty(out_dim, dtype=torch.float))

        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError("Activation must be 'relu' or 'tanh'.")

    def forward(self, x, a=None):
    """
    Apply one graph convolution step.

    Args:
        x: torch.Tensor of shape (n_nodes, in_dim).
            Node feature matrix, where:
            - n_nodes is the number of assets/nodes
            - in_dim is the number of input features per node

        a: torch.Tensor of shape (n_nodes, n_nodes), optional.
            Adjacency matrix. If not provided, uses the fixed adjacency
            stored on the module.

    Returns:
        torch.Tensor of shape (n_nodes, out_dim).
        Transformed node features after adjacency aggregation,
        linear transformation, activation, and bias.
    """

        if a is None:
            if self.a is None:
                raise ValueError('Adjacency matrix must be provided when no fixed adjacency was set.')
            a = self.a
    
        else:
            a = a.to(dtype = self.weight.dtype, device = x.device)
            if a[0,0] == 0:
                a = a + torch.eye(a.shape[0], dtype = a.dtype, device = a.device)

        x = torch.matmul(x, self.weight)
        output = torch.matmul(a, x)
        return self.activation(output) + self.bias


class GCN(BaseModel):
    def __init__(self, n_features, n_pred_per_node, predict=False) -> None:
        """
        Graph convolution model

        Args:
            n_features: int, number of features per node (asset)
            n_pred_per_node: int, number of values to predict for each node (asset)
            predict: bool, does this model directly predict out or is it handled later? If True, creates fully connected layer for prediction
        """
        super().__init__()
        self.gc1 = GraphConv(n_features, n_pred_per_node, 'relu')
        self.predict = predict
        if predict:
            self.fc = nn.Linear(14*n_pred_per_node, 14)

    def forward(self, x, adj=None):
    """
    Run the graph model on one sample.

    Args:
        x: torch.Tensor of shape (n_nodes, n_features) or (1, n_nodes, n_features).
            Node features for all assets.

        adj: torch.Tensor of shape (n_nodes, n_nodes).
            Adjacency matrix describing asset relationships.

    Returns:
        torch.Tensor of shape (1, n_nodes) if predict=True,
        otherwise torch.Tensor of shape (n_nodes, hidden_dim or output_dim),
        depending on the final layer configuration.
    """
        
        x = x.view(1, 14, -1) # reshape so that each node's (asset's) features is own row
        if self.predict:
            return self.fc( self.gc1(x, adj).view(1, -1) )
        else:
            return self.gc1(x, adj)



