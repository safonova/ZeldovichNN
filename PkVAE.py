import os
import sys

import torch
import torch.utils.data
from torch import nn, optim
from torch.utils.data import TensorDataset

import numpy as np
import scipy.linalg

import phate
from sklearn.manifold import TSNE

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm
matplotlib.use("Agg")

from VAE import VAE


def cov_matrix_loss(reconstructed_data, c_sq_inv):
    covariance_estimate = np.cov(reconstructed_data.T)
    precision_mx_estimate = np.linalg.inv(covariance_estimate)
    precision_difference = c_sq_inv @ precision_mx_estimate @ c_sq_inv - np.eye(c_sq_inv.shape[0])
    return np.linalg.norm(precision_difference, 'fro')


def VAE_loss_function(recon_x, x, mu, logvar, c_sq_inv, scales, shifts,
                      KLD_weight=1e-6, grad_weight=0, cov_weight=0):
    x = torch.reshape(x, list(recon_x.shape))
    recon_loss = nn.functional.mse_loss(recon_x, x)
    KLD = torch.sum(-0.5 * (1 + logvar - mu ** 2 - torch.exp(logvar)))
    recon_weight = 1

    gradient = np.gradient(recon_x.detach().numpy(), axis=1)
    abs_sum_grad = torch.sum(torch.Tensor(abs(gradient)/len(recon_x.detach().numpy())))

    covariance_loss = cov_matrix_loss(scales*recon_x.detach().numpy()+shifts, c_sq_inv)

    result = recon_weight * recon_loss + KLD_weight * KLD + abs_sum_grad * grad_weight+covariance_loss*cov_weight

    return result


def train(model,
          train_loader,
          device,
          optimizer,
          KLD_weight,
          grad_weight,
          cov_weight,
          c_sq_inv,
          scales,
          shifts):
    model.train()
    train_loss = 0
    for batch_idx, (data, targets, labels) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = VAE_loss_function(recon_batch,
                                 data,
                                 mu,
                                 logvar,
                                 c_sq_inv,
                                 scales, shifts,
                                 KLD_weight=KLD_weight,
                                 grad_weight=grad_weight,
                                 cov_weight=cov_weight
                                 )
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    return train_loss / len(train_loader.dataset)


def test(model, test_loader, device, KLD_weight, grad_weight, cov_weight, c_sq_inv,  scales, shifts, ):
    test_loss = 0
    with torch.no_grad():
        for data, targets, labels in test_loader:
            data, targets = data.to(device), targets.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += VAE_loss_function(recon_batch, data, mu, logvar, c_sq_inv, scales, shifts,
                                           KLD_weight=KLD_weight,
                                           grad_weight=grad_weight,
                                           cov_weight=cov_weight).item()  # sum up batch loss
    test_loss /= len(test_loader.dataset)

    return test_loss


def visualize_latent(train_latent,
                     test_latent,
                     train_labels,
                     test_labels,
                     train_strings,
                     epoch,
                     args):
    cmap = cm.get_cmap(args.colormap, len(np.unique(train_labels)))
    latent_all = np.vstack([test_latent, train_latent])
    phate_op = phate.PHATE()
    data_phate = phate_op.fit_transform(latent_all)

    fig, axes = plt.subplots(figsize=(12, 8), dpi=120, nrows=2, ncols=3)
    axes[0][0].scatter(
                       data_phate[len(test_latent):, 0],
                       data_phate[len(test_latent):, 1],
                       s=1.5, alpha=0.7,
                       c=train_labels, cmap=cmap)
    axes[0][0].set(title="PHATE of Train latent space",
                   xlabel="PHATE1",
                   ylabel="PHATE2")


    axes[0][1].scatter(
        data_phate[:len(test_latent), 0],
        data_phate[:len(test_latent), 1],
                       s=1.5, alpha=0.7,
                       c=test_labels, cmap=cmap)
    axes[0][1].set(title="PHATE of Test latent space",
                   xlabel="PHATE1",
                   ylabel="PHATE2")

    train_tSNE = TSNE(n_components=2, init='random').fit_transform(train_latent)
    axes[1][0].scatter(train_tSNE[:, 0], train_tSNE[:, 1], s=1.5, alpha=0.7,
                       c=train_labels, cmap=cmap)
    axes[1][0].set(title="tSNE of Train latent space",
                   xlabel="tSNE1",
                   ylabel="tSNE2")

    test_tSNE = TSNE(n_components=2, init='random').fit_transform(test_latent)
    axes[1][1].scatter(test_tSNE[:, 0], test_tSNE[:, 1], s=1.5, alpha=0.7,
                       c=test_labels, cmap=cmap)
    axes[1][1].set(title="tSNE of Test latent space",
                   xlabel="tSNE1",
                   ylabel="tSNE2")
    for ii, cosmostr in enumerate(np.unique(train_strings)):
        axes[0][2].scatter([], [], c=cmap([ii])[0], label=cosmostr)
    axes[0][2].legend(loc='upper left')
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    axes[0][2].axis('off')
    axes[1][2].axis('off')

    plt.savefig(f"{args.savepath}/VAE_latent_epoch_{epoch}.png")


def main(args):
    epochs = args.epochs
    if not os.path.exists(args.savepath):
        os.makedirs(args.savepath)

    with open(args.savepath+"/run_call.txt", 'w') as textfile:
        call_string =" ".join(sys.argv)
        textfile.write(f"{call_string}")

    dataset = np.load(args.datapath)

    train_output = torch.Tensor(dataset['train_output'])
    train_input = torch.Tensor(dataset['train_input'])
    train_labels = torch.Tensor(dataset['train_labels'])
    train_strings = dataset['train_strings']

    test_output = torch.Tensor(dataset['test_output'])
    test_input = torch.Tensor(dataset['test_input'])
    test_labels = torch.Tensor(dataset['test_labels'])
    test_strings = dataset['test_strings']
    scales = dataset['scales']
    shifts = dataset['shifts']

    train_dataset = TensorDataset(train_input, train_output, train_labels)
    test_dataset = TensorDataset(test_input, test_output, test_labels)

    batch_size = args.batch_size
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    covariance = np.load(args.covariancepath)
    c_sq_inv = scipy.linalg.inv(scipy.linalg.sqrtm(covariance))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    torch.manual_seed(args.seed)

    model = VAE(train_input.shape[1], args.latent_width, args.hidden_width).to(device)

    if args.reload:
        model = torch.load(args.savepath + "/checkpt.pth", map_location=device)
        model.eval()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train_losses = []
    test_losses = []
    best_loss = 1e16
    for epoch in range(1, epochs + 1):
        train_loss = train(model,
                           train_loader,
                           device,
                           optimizer,
                           args.KL_weight,
                           args.grad_weight,
                           args.covariance_weight,
                           c_sq_inv,
                           scales, shifts,
                           )
        # Save the latest model state if the loss has decreased
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model, os.path.join(args.savepath, 'checkpt.pth'))

        train_losses.append(train_loss)
        test_loss = test(model, test_loader, device, args.KL_weight,
                         args.grad_weight, args.covariance_weight, c_sq_inv,
                         scales, shifts
                         )
        test_losses.append(test_loss)



        if epoch % args.output_frequency == 0:
            #Write out the current state of the model
            torch.save(model, os.path.join(args.savepath, f'checkpt-{epoch}.pth'))

            train_latent = model.latent_numpy(train_input)
            test_latent = model.latent_numpy(test_input)
            visualize_latent(train_latent,
                             test_latent,
                             train_labels,
                             test_labels,
                             train_strings,
                             epoch,
                             args)

            fig, axes = plt.subplots(figsize=(6,4), dpi=120)
            axes.plot(train_losses, label='Train')
            axes.plot(test_losses, label='Test')
            axes.legend()
            axes.set(xlabel="Epoch", ylabel="Loss")
            plt.savefig(f"{args.savepath}/training_loss_epoch_{epoch}.png")

            print('====> Epoch: {} Average loss: {:.4f}'.format(
                epoch, train_loss / len(train_loader.dataset)))


if __name__=="__main__":
    from parse import parser

    args = parser.parse_args()

    main(args)