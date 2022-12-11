from torch import nn
from torch import Tensor

from linkefl.modelzoo.util import TorchModuleType, make_nn_module


class TabResNet(nn.Module):
    def __init__(self,
                 *,
                 d_in,
                 d_hidden,
                 d_out,
                 n_blocks=2,
                 d_main=None,
                 dropout_first=0.5,
                 dropout_second=0.5,
                 activation: TorchModuleType = 'ReLU',
                 normalization: TorchModuleType = 'BatchNorm1d',
    ) -> None:
        super(TabResNet, self).__init__()
        if d_main is None:
            d_main = d_in

        self.first_layer = nn.Linear(d_in, d_main)
        self.blocks = nn.Sequential(*
            [
                TabResNet.Block(
                    d_main=d_main,
                    d_hidden=d_hidden,
                    bias_first=True,
                    bias_second=True,
                    dropout_first=dropout_first,
                    dropout_second=dropout_second,
                    normalization=normalization,
                    activation=activation,
                    skip_connection=True,
                ) for _ in range(n_blocks)
            ]
        )
        self.head = TabResNet.Head(
            d_in=d_main,
            d_out=d_out,
            bias=True,
            normalization=normalization,
            activation=activation,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.first_layer(x)
        x = self.blocks(x)
        x = self.head(x)
        return x

    class Block(nn.Module):
        """The main building block of TabResNet."""
        def __init__(self,
                     *,
                     d_main: int,
                     d_hidden: int,
                     bias_first: bool,
                     bias_second: bool,
                     dropout_first: float,
                     dropout_second: float,
                     normalization: TorchModuleType,
                     activation: TorchModuleType,
                     skip_connection: bool,
        ) -> None:
            super().__init__()
            self.normalization = make_nn_module(normalization, d_main)
            self.linear_first = nn.Linear(d_main, d_hidden, bias_first)
            self.activation = make_nn_module(activation)
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
        """The final module of TabResNet."""
        def __init__(self,
                     *,
                     d_in: int,
                     d_out: int,
                     bias: bool,
                     normalization: TorchModuleType,
                     activation: TorchModuleType,
        ) -> None:
            super().__init__()
            self.normalization = make_nn_module(normalization, d_in)
            self.activation = make_nn_module(activation)
            self.linear = nn.Linear(d_in, d_out, bias)

        def forward(self, x: Tensor) -> Tensor:
            if self.normalization is not None:
                x = self.normalization(x)
            x = self.activation(x)
            x = self.linear(x)
            return x
