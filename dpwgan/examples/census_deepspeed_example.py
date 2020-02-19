"""
This script generates synthetic data based on ACS PUMS data.
Run `create_census_data.py` to download and create the input data set
before running this script.
"""

import logging
import os

import deepspeed

import pandas as pd
import argparse
# TODO: watch the reqs here, may be screwing up our pandas install ...
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

from dpwgan import CategoricalDataset
from dpwgan.utils import create_categorical_gan

# TODO: or, just pass the fucking thing on the command line ...
# this works in a script ...
THIS_DIR = os.path.dirname(os.path.realpath(__file__))
# ... & this will work ipython if you're trying out the code ...
# THIS_DIR = os.getcwd()
DATA_DIR = os.path.join(THIS_DIR, "data")
CENSUS_FILE = os.path.join(DATA_DIR, 'pums_il.csv')

NOISE_DIM = 20
HIDDEN_DIM = 20
SIGMA = 1

def add_argument():

    parser=argparse.ArgumentParser(description='CIFAR')

    #data
    # cuda
    parser.add_argument('--with_cuda', default=False, action='store_true',
                        help='use CPU in case there\'s no GPU support')
    parser.add_argument('--use_ema', default=False, action='store_true',
                        help='whether use exponential moving average')

    # train
    parser.add_argument('-b', '--batch_size', default=32, type=int,
                        help='mini-batch size (default: 32)')
    parser.add_argument('-e', '--epochs', default=30, type=int,
                        help='number of total epochs (default: 30)')
    parser.add_argument('--local_rank', type=int, default=-1,
                       help='local rank passed from distributed launcher')

    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    return args


def prepare_data(logger):

    logger.info('Preparing data set...')
    try:
        census = pd.read_csv(CENSUS_FILE, dtype=str)
    except FileNotFoundError:
        print('Error: Census data file does not exist.\n'
              'Please run `create_census_data.py` first.')
        return

    census = census.fillna('N/A')
    dataset = CategoricalDataset(census)
    return dataset.to_onehot_flat(), dataset

# TODO: get this thing updated to run on a GPU ...
# TODO: .. or alternately, use the DeepSpeed thing (ZeRO optimisation)

args = add_argument()
torch.manual_seed(123)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

data, dataset = prepare_data(logger)

# TODO: let's just *try* this & see if it works ...
gan = create_categorical_gan(NOISE_DIM, HIDDEN_DIM, dataset.dimensions)
parameters = filter(lambda p: p.requires_grad, gan.parameters())

# Initialize DeepSpeed to use the following features
# 1) Distributed model
# 2) Distributed data loader
# 3) DeepSpeed optimizer

# TODO: note: The engine can wrap any arbitrary model of type torch.nn.module
# TODO: ... well, both the discriminator & the generator are torch.nn - hope
# TODO: that works ...

######################################################################
# To initialize the DeepSpeed engine:
#
# model_engine, optimizer, _, _ = deepspeed.initialize(args=cmd_args,
#                                                      model=model,
#                                                      model_parameters=params)

# deepspeed.intialize ensures that all of the necessary setup required for distributed
# data parallel or mixed precision training are done appropriately under the hood. In
# addition to wrapping the model, DeepSpeed can construct and manage
#
# - the training optimizer
# - data loader, and
# - the learning rate scheduler
#
# based on the parameters passed to deepspeed.initialze and the DeepSpeed configuration file.

# TODO: er, need a config file ...

model_engine, optimizer, trainloader, __ = \
    deepspeed.initialize(args=args,
                         model=gan,
                         model_parameters=parameters,
                         training_data=data)

#######################################################################
# 3. Define a Loss function and optimizer
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Let's use a Classification Cross-Entropy loss and SGD with momentum.




#######################################################################
# TODO: these are the steps you have to go through to ZeROify your
# TODO: network ...
#
# The engine can wrap any arbitrary model of type torch.nn.module and has
# a minimal set of APIs for training and checkpointing the model.
#





def main():
    torch.manual_seed(123)
    # set logging level to INFO to display training
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info('Preparing data set...')
    try:
        census = pd.read_csv(CENSUS_FILE, dtype=str)
    except FileNotFoundError:
        print('Error: Census data file does not exist.\n'
              'Please run `create_census_data.py` first.')
        return
    census = census.fillna('N/A')
    dataset = CategoricalDataset(census)
    data = dataset.to_onehot_flat()

    gan = create_categorical_gan(NOISE_DIM, HIDDEN_DIM, dataset.dimensions)

    logger.info('Training GAN...')
    gan.train(data=data,
              epochs=50,
              n_critics=5,
              learning_rate=1e-4,
              weight_clip=1/HIDDEN_DIM,
              sigma=SIGMA)

    logger.info('Generating synthetic data...')
    flat_synthetic_data = gan.generate(len(census))
    synthetic_data = dataset.from_onehot_flat(flat_synthetic_data)

    filename = os.path.join(THIS_DIR, 'synthetic_pums_il.csv')
    with open(filename, 'w') as f:
        synthetic_data.to_csv(f, index=False)

    logger.info('Synthetic data saved to {}'.format(filename))


#if __name__ == '__main__':
#    main()
