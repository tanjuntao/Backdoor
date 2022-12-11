import torch.random
from torch import nn

from linkefl.modelzoo.util import TorchModuleType, make_nn_module


class MLP(nn.Module):
    def __init__(self,
                 num_nodes,
                 activation: TorchModuleType = 'ReLu',
                 activate_input=False,
                 activate_output=False,
                 random_state=None
        ):
        """
        Parameters
        ----------
        num_nodes
        activation
        activate_input
        activate_output
        random_state
        """
        super(MLP, self).__init__()
        self.activation = make_nn_module(activation)
        if random_state is not None:
            torch.random.manual_seed(random_state)
        modules = []
        n_layers = len(num_nodes) - 1
        for i in range(n_layers):
            modules.append(nn.Linear(num_nodes[i], num_nodes[i+1]))
            modules.append(self.activation)
        if activate_input:
            modules.insert(0, self.activation)
        if not activate_output:
            modules.pop()
        self.sequential = nn.Sequential(*modules)

    def forward(self, x):
        outputs = self.sequential(x)
        return outputs