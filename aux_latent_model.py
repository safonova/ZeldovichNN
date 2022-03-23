from torch import nn
import torch

class AuxLatentNN(nn.Module):
    def __init__(self, input_neurons, model_params, indep_var=None, latent_width=20, hidden_width=400):
        super(VAE, self).__init__()
        self.input_neurons = input_neurons
        self.latent_width = latent_width
        self.hidden_width = hidden_width
        self.model_params = model_params
        self.indep_var = indep_var

        self.encoder = nn.Sequential(nn.Linear(self.input_neurons, self.hidden_width),
                                     nn.LeakyReLU())
        self.predefined = nn.Linear(self.hidden_width, self.model_params)
        self.epsilon = nn.Linear(self.hidden_width, self.latent_width)


        self.decoder = nn.Sequential(nn.Linear(self.latent_width, self.hidden_width),
                                     nn.LeakyReLU(),
                                     nn.Linear(self.hidden_width, self.input_neurons),
                                     nn.Sigmoid()
                                     )

    def std_normal(self, x):
        return torch.exp(-(x ** 2) / 2)

    def deriv_std_norm(self, x):
        return - x * self.std_normal(x)

    def predefined_part(self, latent_params):
        return latent_params[0] * self.indep_var + \
              latent_params[1] * self.std_normal(self.indep_var) + \
              latent_params[2] * self.deriv_std_norm(self.indep_var)


    def forward(self, x):
        encoded1 = self.encoder(x.view(-1, self.input_neurons))
        mu = self.predefined(encoded1)
        eps = self.epsilon()

        y = self.predefined_part(mu)
        z = self.decoder(eps)
        return y + z
