from typing import Type, Union, Callable

import torch.random
from torch import nn
from torch import Tensor

ModuleType = Union[str, Callable[..., nn.Module]]


class MLPModel(nn.Module):
    def __init__(self,
                 num_nodes,
                 activation='relu',
                 activate_input=False,
                 activate_output=False,
                 random_state=None
        ):
        super(MLPModel, self).__init__()
        assert activation in ('relu',), f"{activation} is not supported now."
        if random_state is not None:
            torch.random.manual_seed(random_state)
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


def _make_nn_module(module_type: ModuleType, *args) -> nn.Module:
    if isinstance(module_type, str):
        try:
            cls = getattr(nn, module_type)
        except AttributeError as err:
            raise ValueError(
                    f'Failed to construct the module {module_type} with the arguments {args}'
                ) from err
        return cls(*args)
    else:
        return module_type(*args)


class ResNet(nn.Module):

    class Block(nn.Module):
        """The main building block of `ResNet`."""

        def __init__(
            self,
            *,
            d_main: int,
            d_hidden: int,
            bias_first: bool,
            bias_second: bool,
            dropout_first: float,
            dropout_second: float,
            normalization: ModuleType,
            activation: ModuleType,
            skip_connection: bool,
        ) -> None:
            super().__init__()
            self.normalization = _make_nn_module(normalization, d_main)
            self.linear_first = nn.Linear(d_main, d_hidden, bias_first)
            self.activation = _make_nn_module(activation)
            self.dropout_first = nn.Dropout(dropout_first)
            self.linear_second = nn.Linear(d_hidden, d_main, bias_second)
            self.dropout_second = nn.Dropout(dropout_second)
            self.skip_connection = skip_connection

        def forward(self, x: Tensor) -> Tensor:
            x_input = x
            x = self.normalization(x)
            x = self.linear_first(x)
            x = self.activation(x)
            x = self.dropout_first(x)
            x = self.linear_second(x)
            x = self.dropout_second(x)
            if self.skip_connection:
                x = x_input + x
            return x

    class Head(nn.Module):
        """The final module of `ResNet`."""

        def __init__(
            self,
            *,
            d_in: int,
            d_out: int,
            bias: bool,
            normalization: ModuleType,
            activation: ModuleType,
        ) -> None:
            super().__init__()
            self.normalization = _make_nn_module(normalization, d_in)
            self.activation = _make_nn_module(activation)
            self.linear = nn.Linear(d_in, d_out, bias)

        def forward(self, x: Tensor) -> Tensor:
            if self.normalization is not None:
                x = self.normalization(x)
            x = self.activation(x)
            x = self.linear(x)
            return x

    def __init__(
        self,
        *,
        d_in: int,
        n_blocks: int,
        d_main: int,
        d_hidden: int,
        dropout_first: float,
        dropout_second: float,
        normalization: ModuleType,
        activation: ModuleType,
        d_out: int,
    ) -> None:
        """
        Note:
            `make_baseline` is the recommended constructor.
        """
        super().__init__()

        self.first_layer = nn.Linear(d_in, d_main)
        if d_main is None:
            d_main = d_in
        self.blocks = nn.Sequential(
            *[
                ResNet.Block(
                    d_main=d_main,
                    d_hidden=d_hidden,
                    bias_first=True,
                    bias_second=True,
                    dropout_first=dropout_first,
                    dropout_second=dropout_second,
                    normalization=normalization,
                    activation=activation,
                    skip_connection=True,
                )
                for _ in range(n_blocks)
            ]
        )
        self.head = ResNet.Head(
            d_in=d_main,
            d_out=d_out,
            bias=True,
            normalization=normalization,
            activation=activation,
        )

    # @classmethod
    # def make_baseline(
    #     cls: Type['ResNet'],
    #     *,
    #     d_in: int,
    #     n_blocks: int,
    #     d_main: int,
    #     d_hidden: int,
    #     dropout_first: float,
    #     dropout_second: float,
    #     d_out: int,
    # ) -> 'ResNet':

        # return cls(
        #     d_in=d_in,
        #     n_blocks=n_blocks,
        #     d_main=d_main,
        #     d_hidden=d_hidden,
        #     dropout_first=dropout_first,
        #     dropout_second=dropout_second,
        #     normalization='BatchNorm1d',
        #     activation='ReLU',
        #     d_out=d_out,
        # )

    def forward(self, x: Tensor) -> Tensor:
        x = self.first_layer(x)
        x = self.blocks(x)
        x = self.head(x)
        return x


class CutLayer(nn.Module):
    def __init__(self, in_nodes, out_nodes, random_state=None):
        super(CutLayer, self).__init__()
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes
        if random_state is not None:
            torch.random.manual_seed(random_state)
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
