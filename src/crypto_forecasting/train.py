import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from tqdm import tqdm
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .data import get_crypto_dataset, get_mock_crypto_dataset
from .utils import *
from .components import GCN, LSTM
from .combined_model import AdditiveGraphLSTM, SequentialGraphLSTM


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
FIGURE_DIR = PROJECT_ROOT / "figures"


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # where to perform training


def train(model, dataset, optimizer, criterion, epochs=2, batch_size=1, dl_kws={}, return_all=False, mode='additive'):
    """
    Function that trains a given model on a given dataset using user-defined optimizer/criterion

    Args:
        model: nn.Module, the model to be trained
        dataset: torch Dataset object, contains data for training the model
        optimizer: torch.optim object, controls learning algorithm used for parameter updates
        criterion: function, some loss function to minimize
        epochs: int, number of epochs to train for
        dl_kws: dict, any arguments to pass to DataLoader object
        return_all: bool, for debugging purposes - if True will return all objects to help observe states
        mode: str, whether training is on lstm, gcn, or a combined model (additive or sequential)
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, **dl_kws)
    steps_per_epoch = len(dataloader)
    model.to(device) # send model to desired training device

    model.train()
    epoch_losses = []
    for e in range(epochs):

        print('Starting epoch {}...'.format(e))

        # add tqdm for progress tracking if desired
        epoch_avg_loss = 0
        pbar = tqdm(dataloader) # for produce progress bar to track training
        n_iter = 0
        for features, target, adj in pbar:
            # any casting to correct datatypes here, send to device 
            features, target, adj = features.float().to(device), target.float().to(device), adj.float().to(device)
            
            if mode == 'gcn':
                # special processing, only using last time step
                features = features[:, -1, :]
            else:
                # need to initialize hidden state
                model.initialize_hidden_state(batch_size)

            if mode == 'lstm':
                # lstm only takes in sequence of features
                output, hidden_state = model(features)
                hidden_state = (hidden_state[0].detach(), hidden_state[1].detach())
            else:
                # gcn and combined model both use adjacency matrix
                output = model(features, adj.squeeze())
            loss = criterion(output, target) # make sure this is right order
            
            optimizer.zero_grad()
            loss.backward()
            epoch_avg_loss += loss.item()

            n_iter += 1
            pbar.set_postfix({'avg loss': epoch_avg_loss / n_iter})

            # gradient clipping as necessary
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            optimizer.step()
        epoch_losses.append( epoch_avg_loss / steps_per_epoch )

        # lr_scheduler here if desired

        print('Epoch {} completed. Avg epoch loss: {:.4f}'.format(e, epoch_losses[-1]))
    
    if return_all:
        return model, epoch_losses, optimizer # add any other state-based objects like lr_scheduler here for debugging
    else:
        return model, epoch_losses


def plot_loss(losses, model_name):
    FIGURE_DIR.mkdir("figures", exist_ok = True)

    fig = plt.figure(figsize=(10,5))
    plt.plot(range(len(losses)), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    fig.savefig(str(FIGURE_DIR /f"{model_name}_training_loss.pdf"))


def main(mode, technicals, epochs, model_name, use_mock_data = False):

    print('Creating model...')

    if mode == 'lstm':
        model = LSTM(input_size=98+14*len(technicals), hidden_size=14, batch_first=True, predict=True)
    elif mode == 'gcn':
        model = GCN(n_features=7+len(technicals), n_pred_per_node=3, predict=True) # 7 pre-existing features
    elif mode == 'additive':
        model = AdditiveGraphLSTM(n_features=7+len(technicals), lstm_hidden_dim=14, gcn_pred_per_node=3) # 7 pre-existing features
    else:
        model = SequentialGraphLSTM(n_features=7+len(technicals), lstm_hidden_dim=14, gcn_pred_per_node=3) # 7 pre-existing features
    
    model.float()
    print('Model created.\n')
    print('Creating dataset...')

    if mode == 'gcn':
        seq_len = 1
    else:
        seq_len = 10

    if use_mock_data:
        dataset = get_mock_crypto_dataset(seq_len=seq_len, technicals=technicals, n_samples=100)
    else:
        dataset = get_crypto_dataset(seq_len=seq_len, technicals=technicals)

    optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9) # can play around with this one
    criterion = nn.MSELoss() # regression problem so going just with MSE

    print('Starting training...')
    model, losses = train(model, dataset, optimizer, criterion, epochs=epochs, mode=mode) # change up number of epochs depending on loss plot
    print('Model trained. Saving model...')

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    model.save(str(CHECKPOINT_DIR / f"{model_name}.pth"))

    print('Model saved.')
    plot_loss(losses, model_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_mode', dest='mode', required=True, 
                        choices=['lstm', 'gcn', 'additive', 'sequential'], 
                        help='Which model is going to be trained')
    parser.add_argument('--technicals_config', dest='technicals_config', required=True, 
                        help='json file with mapping of names of features to functions that create feature')
    parser.add_argument('--epochs', dest='epochs', required=True,
                        help='Number of epochs to train model for')
    parser.add_argument('--model_name', dest='model_name', required=True,
                        help='Name for saving model to local directory')
    parser.add_argument('--use_mock_data', action='store_true',
                    help='Use small synthetic dataset instead of Kaggle data')
    args = parser.parse_args()

    with open(args.technicals_config, 'r') as file:
        config = json.load(file)

    for k, v in config.items():
        config[k] = eval(v)

    main(args.mode, config, int(args.epochs), args.model_name, args.use_mock_data)
