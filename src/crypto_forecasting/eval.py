import pandas as pd
import numpy as np
import argparse
import json
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import get_crypto_dataset, get_mock_crypto_dataset
from .utils import *
from .components import GCN, LSTM
from .combined_model import AdditiveGraphLSTM, SequentialGraphLSTM


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # where to perform training


def evaluate(model, dataset, criterion, batch_size=1, dl_kws={}, mode='additive'):
    """
    Function that trains a given model on a given dataset using user-defined optimizer/criterion

    Args:
        model: nn.Module, the model to be trained
        dataset: torch Dataset object, contains data for training the model
        criterion: function, some loss function to minimize
        dl_kws: dict, any arguments to pass to DataLoader object
        mode: str, whether training is on lstm, gcn, or a combined model (additive or sequential)
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, **dl_kws)
    steps_per_epoch = len(dataloader)
    model.to(device) # send model to desired training device

    model.eval()
    losses = []
    for features, target, adj in tqdm(dataloader):
        # any casting to correct datatypes here, send to device 
        features, target, adj = features.float().to(device), target.float().to(device), adj.float().to(device)

        if mode != 'gcn':
            model.initialize_hidden_state(batch_size)

        if mode == 'lstm':
            # lstm only takes in sequence of features
            output, hidden_state = model(features)
            hidden_state = (hidden_state[0].detach(), hidden_state[1].detach())
        else:
            # gcn and combined model both use adjacency matrix
            output = model(features, adj.squeeze())
        loss = criterion(output, target) 
        losses.append(loss.item())
    
    return losses


def main(eval_model, technicals, model_name, use_mock_data = False):

    print('Creating model...')

    if eval_model == 'lstm':
        model = LSTM(input_size=98+14*len(technicals), hidden_size=14, batch_first=True, predict=True)
    elif eval_model == 'gcn':
        model = GCN(n_features=7+len(technicals), n_pred_per_node=3, predict=True) # 7 pre-existing features
    elif eval_model == 'additive':
        model = AdditiveGraphLSTM(n_features=7+len(technicals), lstm_hidden_dim=14, gcn_pred_per_node=3) # 7 pre-existing features
    else:
        model = SequentialGraphLSTM(n_features=7+len(technicals), lstm_hidden_dim=14, gcn_pred_per_node=3) # 7 pre-existing features
    
    model.load('checkpoints/{}.pth'.format(model_name))
    model.float()
    print('Model created.\n')

    print('Creating dataset...')
    if eval_model == 'gcn':
        seq_len = 1
    else:
        seq_len = 10

    if use_mock_data:
        dataset = get_mock_crypto_dataset(seq_len=seq_len, technicals=technicals, evaluation=True, n_samples=30)
    else:
        dataset = get_crypto_dataset(seq_len=seq_len, technicals=technicals, evaluation=True)
    print('Dataset created.\n')

    criterion = nn.MSELoss()
    losses = evaluate(model, dataset, criterion, mode=eval_model)

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / f"{model_name}_loss.txt", "w") as f:
        # output losses to file for later
        for l in losses:
            f.write(f'{l}\n')
    
    print('Average MSE for {}: {:.8f}'.format(model_name, np.mean(losses)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_model', dest='eval_model', required=True, 
                        choices=['lstm', 'gcn', 'additive', 'sequential'], 
                        help='Which model is going to be evaluated')
    parser.add_argument('--technicals_config', dest='technicals_config', required=True, 
                        help='json file with mapping of names of features to functions that create feature')
    parser.add_argument('--model_name', dest='model_name', required=True,
                        help='Name for loading model to local directory')
    parser.add_argument('--use_mock_data', action='store_true',
                    help='Use small synthetic dataset instead of Kaggle data')
    args = parser.parse_args()

    with open(args.technicals_config, 'r') as file:
        config = json.load(file)

    for k, v in config.items():
        config[k] = eval(v)

    main(args.eval_model, config, args.model_name, args.use_mock_data)
