import torch
import torchvision
import argparse
import deepspeed
from torchvision import transforms
from torchvision import datasets
from torch.utils import data
import torch.nn as nn
import logging
import os
import pandas as pd

def add_argument():

    parser=argparse.ArgumentParser(description='CIFAR')

    #data
    # cuda
    parser.add_argument('--with_cuda', default=True, action='store_true',
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


# set up the datasets

transform = transforms.Compose([transforms.ToTensor()])

trainset = datasets.MNIST(root='./data',
                          train=True,
                          download=True,
                          transform=transform)

batch_size = 4

trainloader = data.DataLoader(trainset,
                              batch_size=batch_size,
                              shuffle=True)

testset = datasets.MNIST(root='./data',
                         train=False,
                         download=True,
                         transform=transform)

testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=4,
                                         shuffle=False,
                                         num_workers=2)
def make_some_noise():
    return torch.rand(batch_size, 100)

class generator(nn.Module):

    def __init__(self, inp, out):
        super(generator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(inp, 300),
            nn.ReLU(inplace=True),
            nn.Linear(300, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 800),
            nn.ReLU(inplace=True),
            nn.Linear(800, out)
        )

    def forward(self, x):
        x = self.net(x)
        return x

gen = generator(100, 784)
parameters = filter(lambda p: p.requires_grad, gen.parameters())
args = add_argument()

class discriminator(nn.Module):

    def __init__(self, inp, out):
        super(discriminator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(inp, 300),
            nn.ReLU(inplace=True),
            nn.Linear(300, 300),
            nn.ReLU(inplace=True),
            nn.Linear(300, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, out),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.net(x)
        return x

disc = discriminator(784, 1)

# Initialize DeepSpeed to use the following features
# 1) Distributed model
# 2) Distributed data loader
# 3) DeepSpeed optimizer

# TODO: need a Wasserstein loss

disc_model, disc_optimizer, disc_loader, __ = \
    deepspeed.initialize(args=args,
                         # TODO: JKK: fingers crossed!
                         model=disc,
                         model_parameters=parameters,
                         training_data=trainset)

gen_model, gen_optimizer, gen_loader, __ = \
    deepspeed.initialize(args=args,
                         # TODO: JKK: fingers crossed!
                         model=gen,
                         model_parameters=parameters,
                         training_data=trainset)

criterion = nn.CrossEntropyLoss()

########################################################################
# 4. Train the network
# ^^^^^^^^^^^^^^^^^^^^
#
# This is when things start to get interesting.
# We simply have to loop over our data iterator, and feed the inputs to the
# network and optimize.


# fake_inputs = torch.randn(batch_size, 100)

for epoch in range(20):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader):
        # forward() method
        # loss = disc_engine(batch)
        # real data
        real_inputs, real_labels = data[0].to(disc_model.local_rank), data[1].to(disc_model.local_rank)
        real_outputs = disc_model(real_inputs)
        real_loss = criterion(real_outputs, real_labels)

        # fake data


