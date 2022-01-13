import argparse

parser = argparse.ArgumentParser(description="VAE training parameters")

parser.add_argument("--savepath", type=str, default="../results/tmp",
                    help='Give the path for saving training output, relative to the directory where the code is running.')
parser.add_argument("--latent_width", type=int, default=20, help='Number of nuerons in latent space')
parser.add_argument("--batch_size", type=int, default=25, help='Select batch size')
parser.add_argument('--datapath', type=str,
                    help='Path to the dataset dictionary, containing train set, test set, labels, etc.')
parser.add_argument("--output_frequency", type=int, default=200,
                    help='How many epochs should pass between saving model output and visualizing')
parser.add_argument("--reload", action="store_true",
                    help="Resurrects existing model from last checkpoint in the savepath.")
parser.add_argument("--KL_weight", type=float, default=1,
                    help='Weight applied to the KL divergence term int he loss function.')
parser.add_argument("--epochs", type=int, default=10000, help='Number of epochs to train over.')
parser.add_argument("--lr", type=float, default=5e-7, help='Learning rate for the optimizer.')
parser.add_argument("--seed", type=int, default=1, help="Seed for random processes in PyTorch.")
parser.add_argument("--no_cuda", action="store_true", help="Makes the script on a CPU even if a GPU is available.")
parser.add_argument("--hidden_width", type=int, default=400,
                    help="Number of neurons in hidden layer for a network with 1 hidden layer.")