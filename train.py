import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from trainer import Trainer
from model.AutoEncoder import Model_selector

from data_loader import get_loader

def define_argparser():

    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', default='Autoencoder_deep_layer')
    p.add_argument('--model_name', type=str, default='Autoencoder_deep_layer')
    p.add_argument('--activation', type=str, default='ReLU')
    p.add_argument('--noise', type=bool, default=True)
    p.add_argument('--train_ratio', type=float, default=.8)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--n_epochs',type=int, default=500)
    p.add_argument('--verbose', type=int, default=2)

    config = p.parse_args()

    return config

def main(config=None):

    device = torch.device('cpu')

    train_loader, valid_loader = get_loader()

    print('Train:', len(train_loader.dataset))
    print('Valid:', len(valid_loader.dataset))

    # Just for extract the size of data
    size = next(iter(train_loader))[0][0].size(0)

    model = Model_selector(size, config.model_name, config.activation, config.noise)

    optimizer = optim.Adam(model.parameters())
    crit = nn.MSELoss()

    trainer = Trainer(config)
    trainer.train(model, crit, optimizer, train_loader, valid_loader)

if __name__ == '__main__':

    config = define_argparser()
    main(config)
