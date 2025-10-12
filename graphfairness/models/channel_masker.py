import torch

class channel_masker(torch.nn.Module):
    def __init__(self, num_features):
        super(channel_masker, self).__init__()

        self.weights = torch.nn.Parameter(torch.distributions.Uniform(
            0, 1).sample((num_features, 2)))

    def reset_parameters(self):
        self.weights = torch.nn.init.xavier_uniform_(self.weights)

    def forward(self):
        return self.weights