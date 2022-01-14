from torch import nn
import torch

class VAE(nn.Module):
    def __init__(self, input_neurons, latent_width=20, hidden_width=400):
        super(VAE, self).__init__()
        self.input_neurons = input_neurons
        self.latent_width = latent_width
        self.hidden_width = hidden_width

        self.encoder = nn.Sequential(nn.Linear(self.input_neurons, self.hidden_width),
                                     nn.PReLU()
                                     )
        '''self.fc1 = nn.Linear(self.input_neurons, self.hidden_width)'''
        self.fc21 = nn.Linear(self.hidden_width, self.latent_width)
        self.fc22 = nn.Linear(self.hidden_width, self.latent_width)
        '''self.fc3 = nn.Linear(self.latent_width, self.hidden_width)
        self.fc4 = nn.Linear(self.hidden_width, self.input_neurons)
        '''

        self.decoder = nn.Sequential(nn.Linear(self.latent_width, self.hidden_width),
                                     nn.PReLU(),
                                     nn.Linear(self.hidden_width, self.input_neurons),
                                     nn.Sigmoid()
                                     )

    def encode(self, x):
        '''m = nn.PReLU()
        h1 = m(self.fc1(x))'''
        h1 = self.encoder(x)
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        '''m = nn.PReLU()
        h3 = m(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))'''
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_neurons))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def latent_numpy(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_neurons))
        z = self.reparameterize(mu, logvar)
        latent = z.detach().numpy()
        return latent
