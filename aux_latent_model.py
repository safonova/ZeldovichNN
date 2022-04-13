import h5py
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.utils.data
from torch import nn, optim
from torch.utils.data import TensorDataset
import math
from scipy.interpolate import interp1d
import pandas as pd

class AuxLatentNN(nn.Module):
    def __init__(self, input_neurons, model_params, xi_template, indep_var=None, latent_width=20, hidden_width=400):
        super(AuxLatentNN, self).__init__()
        self.input_neurons = input_neurons
        self.latent_width = latent_width
        self.hidden_width = hidden_width
        self.model_params = model_params
        self.indep_var = indep_var
        self.xi_template = xi_template

        self.encoder = nn.Sequential(nn.Linear(self.input_neurons,
                                               self.hidden_width),
                                     nn.LeakyReLU(),
                                     nn.Linear(self.hidden_width,
                                               2 * self.hidden_width),
                                     nn.LeakyReLU(),
                                     )
        self.predefined = nn.Sequential(nn.Linear(2 * self.hidden_width,
                                                  self.model_params),
                                        nn.Sigmoid()
                                        )
        self.epsilon = nn.Sequential(nn.Linear(2 * self.hidden_width,
                                               self.latent_width))

        self.decoder = nn.Sequential(CustomIntermediateLayer(self.latent_width,
                                                             2 * self.hidden_width),
                                     CustomIntermediateLayer(2 * self.hidden_width,
                                                             self.hidden_width),
                                     CustomFinalLayer(self.hidden_width,
                                                      self.input_neurons,
                                                      self.xi_template,
                                                      self.indep_var),
                                     nn.Sigmoid()
                                     )

    def forward(self, x):
        encoded = self.encoder(x)
        alpha = self.predefined(encoded)
        eps = self.epsilon(encoded)
        result = self.decoder([eps, alpha])
        # result = result.reshape(result.shape[1:])
        return result, alpha, eps


class CustomFinalLayer(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer with modifications"""

    def __init__(self, size_in, size_out, xi_template, r):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        weights = torch.Tensor(size_out, size_in)
        self.weights = nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter.
        bias = torch.Tensor(size_out)
        self.bias = nn.Parameter(bias)

        # For the custom model
        self.xi_template = xi_template
        self.r = r
        df = pd.read_csv("xi_scaling_polynomial.txt")
        self.width_params = df['coefficient'][np.where(df['type'] == "width")[0]].values
        self.width_degrees = df['degree'][np.where(df['type'] == "width")[0]].values
        self.mid_params = df['coefficient'][np.where(df['type'] == "middle")[0]].values
        self.mid_degrees = df['degree'][np.where(df['type'] == "middle")[0]].values

        # initialize weights and biases
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))  # weight init
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

    def reparametrize_xi_inside_NN(self, xi, r):
        width_poly = 0
        for ii, param in enumerate(self.width_params):
            width_poly += param * r.detach().numpy() ** self.width_degrees[ii]
        middle_poly = 0
        for ii, param in enumerate(self.mid_params):
            middle_poly += param * r.detach().numpy() ** self.mid_degrees[ii]
        xi_polynom = (xi - middle_poly) / (1.25 * width_poly) + 0.5
        return xi_polynom.flatten()

    def undo_alpha_prime(self, alpha_prime):
        alpha = 0.1 * alpha_prime + 0.95
        return alpha

    def forward(self, packaged_input):
        x, alpha_prime = packaged_input
        alpha = self.undo_alpha_prime(alpha_prime)
        template_xi_result = np.array([self.xi_template(aleph * self.r.detach().numpy())[0]
                                       for aleph in alpha.detach().numpy()])

        try:
            rescaled_xi_result = torch.Tensor([self.reparametrize_xi_inside_NN(res, self.r)
                                               for res in template_xi_result])
        except ValueError as e:
            print(e)
            print(template_xi_result)
        w_times_x = torch.mm(x, self.weights.t())
        linear_layer_result = torch.add(w_times_x, self.bias)

        total_result = torch.add(linear_layer_result, rescaled_xi_result)
        return total_result


class CustomIntermediateLayer(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer with modifications"""

    def __init__(self, size_in, size_out, ):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        weights = torch.Tensor(size_out, size_in)
        self.weights = nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter.
        bias = torch.Tensor(size_out)
        self.bias = nn.Parameter(bias)
        # initialize weights and biases
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))  # weight init
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

    def forward(self, packaged_input):
        x, latent_params = packaged_input
        w_times_x = torch.mm(x, self.weights.t())
        result = torch.add(w_times_x, self.bias)
        m = nn.LeakyReLU()
        result_activated = m(result)
        return result_activated, latent_params

def reparametrize_xi(xi, r):
    df = pd.read_csv("xi_scaling_polynomial.txt")
    width_params = df['coefficient'][np.where(df['type']=="width")[0]]
    width_degrees = df['degree'][np.where(df['type']=="width")[0]]
    mid_params = df['coefficient'][np.where(df['type']=="middle")[0]]
    mid_degrees = df['degree'][np.where(df['type']=="middle")[0]].values
    width_poly = 0
    for ii, param in enumerate(width_params):
        width_poly += param * r ** width_degrees[ii]

    middle_poly = 0
    for ii, param in enumerate(mid_params):
        middle_poly += param * r ** mid_degrees[ii]

    xi_polynom = (xi-middle_poly)/(1.25* width_poly)+0.5
    return xi_polynom

def reparametrize_alpha(alpha):
    alpha_prime = (alpha-1)*10+0.5
    return alpha_prime

def undo_alpha_prime(alpha_prime):
    alpha = 0.1 * alpha_prime + 0.95
    return alpha


def train(model, train_loader, device, optimizer, inv_covariance, template_fn, indep_var):
    model.train()
    train_loss = 0
    for batch_idx, (data, alpha_prime) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, eps = model(data)
        loss = mixed_loss_function(recon_batch,
                                   data, mu,
                                   alpha_prime,
                                   inv_covariance,
                                   template_fn,
                                   indep_var
                                   )
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    return train_loss / len(train_loader.dataset)


def test(model, test_loader, device, inv_covariance, template_fn, indep_var):
    test_loss_recon_eps = 0
    test_loss_recon_temp = 0
    test_loss_param = 0

    with torch.no_grad():
        for data, alpha_true in test_loader:
            data, alpha_true = data.to(device), alpha_true.to(device)
            recon_batch, alpha_fit, eps = model(data)
            batch_loss_recon_eps, batch_loss_recon_temp, batch_loss_param = mixed_loss_function(recon_batch,
                                                                                                data,
                                                                                                alpha_fit,
                                                                                                alpha_true,
                                                                                                inv_covariance,
                                                                                                template_fn,
                                                                                                indep_var,
                                                                                                separated=True)  # sum up batch loss
            test_loss_recon_eps += batch_loss_recon_eps.item()
            test_loss_recon_temp += batch_loss_recon_temp.item()
            test_loss_param += batch_loss_param.item()
    test_loss = (test_loss_recon_eps + test_loss_recon_temp + test_loss_param) / len(test_loader.dataset)
    test_loss_recon_eps /= len(test_loader.dataset)
    test_loss_recon_temp /= len(test_loader.dataset)
    test_loss_param /= len(test_loader.dataset)
    return test_loss, test_loss_recon_eps, test_loss_recon_temp, test_loss_param


def mixed_loss_function(recon_x, x, alpha_prime, true_alpha,
                        inv_covariance, template_fn, indep_var,
                        separated=False):
    alpha_prime = alpha_prime.reshape(true_alpha.shape)
    recon_x = recon_x.reshape(x.shape)
    template_model_fit = torch.Tensor([reparametrize_xi(template_fn(indep_var * undo_alpha_prime(al)), indep_var)
                                       for al in alpha_prime.detach().numpy()])
    template_true_fit = torch.Tensor([reparametrize_xi(template_fn(indep_var * undo_alpha_prime(al)), indep_var)
                                      for al in true_alpha.detach().numpy()])

    epsilon_model = recon_x - template_model_fit

    epsilon_true = x - template_true_fit
    '''chisq = torch.sum(torch.Tensor([recon_difference.T #@ torch.Tensor(inv_covariance)  
                                    @ recon_difference 
                                    for recon_difference in  eps_diff]))'''
    recon_eps = nn.functional.mse_loss(epsilon_model, epsilon_true, reduction='mean')
    recon_template = nn.functional.mse_loss(template_model_fit, template_true_fit, reduction='mean') * 5e1

    param_loss = nn.functional.mse_loss(alpha_prime, true_alpha)
    if separated:
        return recon_eps, recon_template, param_loss
    else:
        return recon_eps + recon_template + param_loss



def main(args):
    data = h5py.File("vae-fit-test0.h5", 'r')

    alpha = data['alpha'][...]
    cov = data['cov'][...]
    r = data['r'][...]
    r0 = data['r0'][...]
    xi = data['xi'][...]
    xi0 = data['xi0'][...]


    xi_prime = reparametrize_xi(xi, r)
    alpha_prime = reparametrize_alpha(alpha)
    xi_template = interp1d(r0, xi0)

    Ntrain = int(len(xi_prime)*0.8)
    train_input = torch.Tensor(xi_prime[:Ntrain,:])
    test_input = torch.Tensor(xi_prime[Ntrain:,:])
    train_dataset = TensorDataset(train_input, torch.Tensor(alpha_prime[:Ntrain]))
    test_dataset = TensorDataset(test_input, torch.Tensor(alpha_prime[Ntrain:]))

    batch_size = args.batch_size
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=True)

    cudabool = torch.cuda.is_available()
    device = torch.device("cuda" if cudabool else "cpu")

    torch.manual_seed(1)

    model = AuxLatentNN(train_input.shape[1],
                        1, xi_template,
                        indep_var=torch.Tensor([r]).reshape([1, 62]),
                        latent_width=args.latent_width, hidden_width=args.hidden_width).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    train_losses = []
    test_losses = []
    test_losses_recon_eps = []
    test_losses_recon_temp = []
    test_losses_param = []
    best_loss = 1e16
    for epoch_cluster in range(int(args.epochs/args.output_frequency)):
        for epoch in range(args.output_frequency):
            loss = train(model, train_loader, device,
                         optimizer, np.linalg.inv(cov), xi_template, r)
            train_losses.append(loss)
            loss_total, loss_recon_eps, loss_recon_temp, loss_param = test(model, test_loader, device,
                                                                           np.linalg.inv(cov), xi_template, r)
            test_losses.append(loss_total)
            test_losses_recon_eps.append(loss_recon_eps)
            test_losses_recon_temp.append(loss_recon_temp)
            test_losses_param.append(loss_param)
            print(len(train_losses))
            if loss_total < best_loss:
                best_loss = loss_total
                torch.save(model, args.savepath + 'checkpt.pth')

        fig, axes = plt.subplots(dpi=120)
        plt.plot(train_losses, label='train')
        plt.plot(test_losses, label='test')
        plt.legend(loc='upper right')
        plt.yscale('log')
        plt.savefig(f"{args.savepath}losses_epoch_{(1+epoch_cluster) * epoch}")

        fig, axes = plt.subplots(dpi=120)
        plt.plot(test_losses_recon_eps, label='epsilon reconstruction loss')
        plt.plot(test_losses_recon_temp, label="template reconstruction loss")
        plt.plot(test_losses_param, label='parameter loss')
        plt.legend(loc='upper right')
        plt.yscale('log')
        plt.savefig(f"{args.savepath}loss_components_epoch_{(1+epoch_cluster) * epoch}")

        encoded1 = model.encoder(test_input.view(-1, 62))
        alpha_fit = model.predefined(encoded1)

        fig, axes = plt.subplots(figsize=(6, 5), dpi=120)
        plt.hist(undo_alpha_prime(alpha_fit.detach().numpy().flatten()) -
                 undo_alpha_prime(alpha_prime[Ntrain:]), np.linspace(-0.04, 0.04, 20))
        plt.xlabel(r"$\alpha$ NN fit - truth")
        plt.savefig(f"{args.savepath}alpha_histogram_epoch_{(1+epoch_cluster) * epoch}")


        recon_batch, mu, eps = model(test_input)
        template_model_fit = torch.Tensor([reparametrize_xi(xi_template(r * undo_alpha_prime(al)), r)
                                           for al in mu.detach().numpy()])
        template_true_fit = torch.Tensor([reparametrize_xi(xi_template(r * al), r)
                                          for al
                                          in alpha[Ntrain:]])

        epsilon_model = recon_batch - template_model_fit
        epsilon_true = torch.Tensor(xi_prime[Ntrain:]) - template_true_fit

        idx = np.random.choice(len(test_input) - 4)

        fig, axes = plt.subplots(figsize=(6, 4), dpi=120)
        for ii in range(idx, idx + 3):
            plt.plot((xi_prime[Ntrain + ii]),
                     c=f"C0{ii}", ls=":")
            plt.plot(recon_batch.detach().numpy()[ii], ls="-.",
                     c=f"C0{ii}")
        plt.plot([], c="k", label="Ground truth", ls=":")
        plt.plot([], c="k", label="Reconstruction", ls="-.")
        plt.legend()
        plt.savefig(f"{args.savepath}xi_reconstruction_epoch_{(1+epoch_cluster) * epoch}")

        fig, axes = plt.subplots(figsize=(6, 4), dpi=120)
        for ii in range(idx, idx + 3):
            plt.plot(template_true_fit.detach().numpy()[ii] - template_model_fit.detach().numpy()[ii], ls="-",
                     c=f"C0{ii}", label=r"$\alpha$" + f"{undo_alpha_prime(mu.detach().numpy())[ii][0]}")
        plt.legend(loc='upper left')
        plt.title("True template - fit template")

        plt.savefig(f"{args.savepath}template_difference_epoch_{(1+epoch_cluster) * epoch}")

        fig, axes = plt.subplots(figsize=(6, 4), dpi=120)
        for ii in range(idx, idx + 3):
            plt.plot(template_true_fit.detach().numpy()[ii],
                     c=f"C0{ii}", ls=":")
            plt.plot(template_model_fit.detach().numpy()[ii], ls="-.",
                     c=f"C0{ii}")
        plt.plot([], c="k", label="Ground truth", ls=":")
        plt.plot([], c="k", label="Reconstruction", ls="-.")
        plt.legend(loc='upper left')
        plt.title("Model")
        plt.savefig(f"{args.savepath}template_epoch_{epoch_cluster * epoch}")

        fig, axes = plt.subplots(figsize=(6, 4), dpi=120)
        for ii in range(idx, idx + 3):
            plt.plot(epsilon_true.detach().numpy()[ii],
                     c=f"C0{ii}", ls=":")
            plt.plot(epsilon_model.detach().numpy()[ii], ls="-.",
                     c=f"C0{ii}")
        plt.plot([], c="k", label="Ground truth", ls=":")
        plt.plot([], c="k", label="Reconstruction", ls="-.")
        plt.legend(loc='upper left')
        plt.title("Epsilon")
        plt.savefig(f"{args.savepath}epsilon_reconstruction_epoch_{(1+epoch_cluster) * epoch}")

if __name__ == "__main__":
    from parse import parser
    args = parser.parse_args()

    main(args)