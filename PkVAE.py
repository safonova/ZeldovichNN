import os
import sys

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset

import numpy as np

import phate
from sklearn.preprocessing import scale
from sklearn.manifold import TSNE

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm
matplotlib.use("Agg")


class VAE(nn.Module):
    def __init__(self, input_neurons, latent_width=20, hidden_width=400):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(input_neurons, 400)
        self.fc21 = nn.Linear(400, latent_width)
        self.fc22 = nn.Linear(400, latent_width)
        self.fc3 = nn.Linear(latent_width, 400)
        self.fc4 = nn.Linear(400, input_neurons)

    def encode(self, x):
        m = nn.PReLU()
        h1 = m(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        m = nn.PReLU()
        h3 = m(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, ks_to_keep))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def latent_numpy(self, x):
        mu, logvar = self.encode(x.view(-1, ks_to_keep))
        z = self.reparameterize(mu, logvar)
        latent = z.detach().numpy()
        return latent


def VAE_loss_function(recon_x, x, mu, logvar, KLD_weight=1e-6):
    x = torch.reshape(x, list(recon_x.shape))
    recon_loss = nn.functional.mse_loss(recon_x, x)
    KLD = torch.sum(-0.5 * (1 + logvar - mu ** 2 - torch.exp(logvar)))
    recon_weight = 1
    result = recon_weight * recon_loss + KLD_weight * KLD
    return result


def train(model, train_loader, device, optimizer):
    model.train()
    train_loss = 0
    for batch_idx, (data, targets, labels) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = VAE_loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    return train_loss / len(train_loader.dataset)


def main(args):
    epochs = args.epochs
    if not os.path.exists(args.savepath):
        os.makedirs(args.savepath)

    with open(savepath+"/run_call.txt", 'w') as textfile:
        call_string =" ".join(sys.argv)
        textfile.write(f"{call_string}")

    dataset = np.load(args.datapath)

    train_output = dataset['train_output']
    train_input = dataset['train_input']
    train_labels = dataset['train_labels']

    test_output = dataset['test_output']
    test_input = dataset['test_input']
    test_labels = dataset['test_labels']

    train_dataset = TensorDataset(train_input, train_output, train_labels)
    test_dataset = TensorDataset(test_input, test_output, test_labels)

    batch_size = args.batch_size
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    model = VAE(train_input.shape[1], args.latent_width)
    losses = []
    for epoch in range(1, epochs + 1):
        train_loss = train(epoch, model)
        losses.append(train_loss)
        if epoch % 200 == 0:
            train_latent = model.latent_numpy(train_input)
            phate_op = phate.PHATE()
            data_phate = phate_op.fit_transform(train_latent)

            fig, axes = plt.subplots(figsize=(8, 8), dpi=120, nrows=2, ncols=2)
            axes[0][0].scatter(data_phate[:, 0], data_phate[:, 1], s=1.5, alpha=0.7, c=train_labels)
            axes[0][0].set(title="PHATE of Train latent space",
                        xlabel="PHATE1",
                        ylabel="PHATE2")

            test_latent = model.latent_numpy(test_input)
            phate_op = phate.PHATE()
            data_phate = phate_op.fit_transform(test_latent)

            s1 = axes[0][1].scatter(data_phate[:, 0], data_phate[:, 1], s=1.5, alpha=0.7, c=test_labels)
            axes[1][1].set(title="PHATE of Test latent space",
                        xlabel="PHATE1",
                        ylabel="PHATE2")
            print('====> Epoch: {} Average loss: {:.4f}'.format(
                epoch, train_loss / len(train_loader.dataset)))
            plt.colorbar(s1)

            train_tSNE = TSNE(n_components=2, learning_rate='auto',init = 'random').fit_transform(train_latent)
            axes[1][0].scatter(train_tSNE[:, 0], train_tSNE[:, 1], s=1.5, alpha=0.7, c=train_labels)
            axes[1][0].set(title="tSNE of Train latent space",
                        xlabel="tSNE1",
                        ylabel="tSNE2")

            test_tSNE = TSNE(n_components=2, learning_rate='auto',init = 'random').fit_transform(test_latent)
            axes[1][1].scatter(test_tSNE[:, 0], test_tSNE[:, 1], s=1.5, alpha=0.7, c=test_labels)
            axes[1][1].set(title="tSNE of Test latent space",
                        xlabel="tSNE1",
                        ylabel="tSNE2")

            plt.savefig(f"{args.savepath}VAE_latent_epoch_{epoch}.png")

if __name__=="__main__":
    from parse import parser

    args = parser.parse_args()
    savepath = args.save

    main(args)