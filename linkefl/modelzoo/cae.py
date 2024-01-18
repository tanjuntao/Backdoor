import torch
import torch.nn as nn


def initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal(m.weight.data)
        # if m.bias is not None:
        #     m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight.data, 0, 0.01)
            m.bias.data.zero_()


def sharpen(probabilities, T):
    if probabilities.ndim == 1:
        # print("here 1")
        tempered = torch.pow(probabilities, 1 / T)
        tempered = tempered / (torch.pow((1 - probabilities), 1 / T) + tempered)
    else:
        # print("here 2")
        tempered = torch.pow(probabilities, 1 / T)
        tempered = tempered / tempered.sum(dim=-1, keepdim=True)

    return tempered


class AutoEncoder(nn.Module):
    def __init__(self, num_classes):
        input_dim = num_classes
        encode_dim = (2 + 6 * num_classes) ** 2
        super(AutoEncoder, self).__init__()
        self.d = input_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encode_dim**2),
            nn.ReLU(),
            nn.Linear(encode_dim**2, input_dim),
            nn.Softmax(dim=1),
        )
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, encode_dim**2),
            nn.ReLU(),
            nn.Linear(encode_dim**2, input_dim),
            nn.Softmax(dim=1),
        )
        initialize_weights(self)

    def decode(self, d_y):
        return self.decoder(d_y)

    def forward(self, x):
        z = self.encoder(x.view(-1, self.d))
        # d_y = F.softmax(z, dim=1)
        d_y = sharpen(z, T=1.0)
        return self.decoder(d_y), d_y
        # return self.decoder(z), d_y

    def load_model(self, model_full_name, target_device="cuda:0"):
        self.load_state_dict(torch.load(model_full_name, map_location=target_device))

    def save_model(self, model_full_name):
        torch.save(self.state_dict(), model_full_name)
