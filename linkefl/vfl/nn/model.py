from torch import nn


class MLPModel(nn.Module):
    def __init__(self,
                 num_nodes,
                 activation='relu',
                 activate_input=False,
                 activate_output=False
        ):
        super(MLPModel, self).__init__()
        assert activation in ('relu',), f"{activation} is not supported now."
        modules = []
        n_layers = len(num_nodes) - 1
        for i in range(n_layers):
            modules.append(nn.Linear(num_nodes[i], num_nodes[i+1]))
            modules.append(nn.ReLU())
        if activate_input:
            modules.insert(0, nn.ReLU())
        if not activate_output:
            modules.pop()
        self.sequential = nn.Sequential(*modules)

    def forward(self, x):
        outputs = self.sequential(x)
        return outputs


class CutLayer(nn.Module):
    def __init__(self, in_nodes, out_nodes):
        super(CutLayer, self).__init__()
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes
        self.linear = nn.Linear(in_nodes, out_nodes) # no activation

    def forward(self, x):
        outputs = self.linear(x)
        return outputs


class BottomModel(nn.Module):
    """Bottom model base class."""
    def __init__(self, num_nodes: list):
        """Initialize model.

        Args:
            num_nodes[List]: number of neurons of each layer of RSAPSIPassive's MLP model.
        """
        super(BottomModel, self).__init__()

        modules = []
        for idx in range(len(num_nodes) - 1):
            modules.append(nn.Linear(num_nodes[idx], num_nodes[idx + 1]))
            modules.append(nn.ReLU())
        self.sequential = nn.Sequential(*modules)

    def forward(self, x):
        outputs = self.sequential(x)
        return outputs


class PassiveBottomModel(BottomModel):
    """RSAPSIPassive bottom model arthitecture."""
    def __init__(self, num_nodes: list):
        super(PassiveBottomModel, self).__init__(num_nodes)

    def forward(self, x):
        # x_flat = self.flatten(x)[:, :int(28*28/2)]
        return super(PassiveBottomModel, self).forward(x)


class ActiveBottomModel(BottomModel):
    """RSAPSIActive bottom model architecture."""
    def __init__(self, num_nodes: list):
        super(ActiveBottomModel, self).__init__(num_nodes)

    def forward(self, x):
        # x_flat = self.flatten(x)[:, int(28*28/2):]
        return super(ActiveBottomModel, self).forward(x)


class IntersectionModel(nn.Module):
    """Intersection model arthitecture."""
    def __init__(self, num_nodes):
        """Initialize intersection model.

        Args:
            num_nodes[List]:
                First item: input from RSAPSIPassive;
                Second item: input from RSAPSIActive;
                Third item: output of intersection layer
        """
        super(IntersectionModel, self).__init__()
        alice_input_dim, bob_input_dim, output_dim = num_nodes
        input_dim = alice_input_dim + bob_input_dim
        modules = [nn.Linear(input_dim, output_dim), nn.ReLU()]
        self.sequential = nn.Sequential(*modules)

    def forward(self, x):
        outputs = self.sequential(x)
        return outputs


class TopModel(nn.Module):
    """Top model arthitecture."""
    def __init__(self, num_nodes):
        super(TopModel, self).__init__()
        modules = []
        for idx in range(len(num_nodes) - 1):
            modules.append(nn.Linear(num_nodes[idx], num_nodes[idx + 1]))
            if idx != len(num_nodes) - 2:
                modules.append(nn.ReLU())
        self.sequential = nn.Sequential(*modules)

    def forward(self, x):
        logits = self.sequential(x)
        return logits


class SubstituteModel(TopModel):
    """RSAPSIPassive's local substitute model architecture."""
    def __init__(self, num_nodes, alice_bottom_model, fine_tuning=False):
        super(SubstituteModel, self).__init__(num_nodes)
        self.alice_bottom_model = alice_bottom_model
        if not fine_tuning:
            for param in self.alice_bottom_model.parameters():
                param.requires_grad = False

    def forward(self, x):
        inter = self.alice_bottom_model(x)
        logits = self.sequential(inter)
        return logits
