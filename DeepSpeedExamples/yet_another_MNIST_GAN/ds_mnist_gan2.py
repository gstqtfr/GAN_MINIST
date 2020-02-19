import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
import datetime
import os, sys
#from matplotlib.pyplot import imshow, imsave

from dpwgan import DPWGAN, MultiCategoryGumbelSoftmax

import pandas as pd
import torch
import logging

from dpwgan import CategoricalDataset
from dpwgan.utils import create_categorical_gan, percentage_crosstab
from torch.autograd import Variable


def create_categorical_generator(noise_dim, hidden_dim, output_dims):
    generator = torch.nn.Sequential(
        torch.nn.Linear(noise_dim, hidden_dim),
        torch.nn.ReLU(),
        MultiCategoryGumbelSoftmax(hidden_dim, output_dims)
    )
    return generator

def create_categorical_discriminator(noise_dim, hidden_dim, output_dims):
    discriminator = torch.nn.Sequential(
        torch.nn.Linear(sum(output_dims), hidden_dim),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(hidden_dim, 1)
    )
    return discriminator

def noise_function(n, noise_dim):
    return torch.randn(n, noise_dim)

def generate(n, noise_dim, generator):
        """Generate a synthetic data set using the trained model

        Parameters
        ----------
        n : int
            Number of data points to generate

        Returns
        -------
        torch.Tensor
        """
        noise = noise_function(n, noise_dim)
        fake_sample = generator(noise)
        return fake_sample

# TODO: JKK: let's make this *really* simple ...


NOISE_DIM = 10
HIDDEN_DIM = 20
SIGMA = 1
learning_rate = 1e-3
epochs = 500
n_critics = 5
batch_size = 128
weight_clip = 1/HIDDEN_DIM
sigma = SIGMA

def generate_data():
    df = pd.DataFrame(
        {'weather': ['sunny']*10000+['cloudy']*10000+['rainy']*10000,
         'status': ['on time']*8000+['delayed']*2000
         + ['on time']*3000+['delayed']*5000+['canceled']*2000
         + ['on time']*2000+['delayed']*4000+['canceled']*4000}
    )
    return df


torch.manual_seed(123)
logger = logging.getLogger('spam_application')
logging.basicConfig(level=logging.INFO)

real_data = generate_data()
dataset = CategoricalDataset(real_data)
data_tensor = dataset.to_onehot_flat()

generator = create_categorical_generator(NOISE_DIM, HIDDEN_DIM, dataset.dimensions)
discriminator = create_categorical_discriminator(NOISE_DIM, HIDDEN_DIM, dataset.dimensions)

generator_solver = torch.optim.RMSprop(
    generator.parameters(), lr=learning_rate
)

discriminator_solver = torch.optim.RMSprop(
    discriminator.parameters(), lr=learning_rate
)

epoch_length = len(real_data) / (n_critics * batch_size)
n_iters = int(epochs * epoch_length)

for iteration in range(n_iters):
    for _ in range(n_critics):
        # Sample real data
        rand_perm = torch.randperm(data_tensor.size(0))
        samples = data_tensor[rand_perm[:batch_size]]
        real_sample = Variable(samples)

        # Sample fake data
        fake_sample = generate(batch_size, NOISE_DIM, generator)

        # Score data
        discriminator_real = discriminator(real_sample)
        discriminator_fake = discriminator(fake_sample)

        # Calculate discriminator loss
        # Discriminator wants to assign a high score to real data
        # and a low score to fake data
        discriminator_loss = -(
                torch.mean(discriminator_real) -
                torch.mean(discriminator_fake)
        )

        discriminator_loss.backward()
        discriminator_solver.step()

        # Weight clipping for privacy guarantee
        for param in discriminator.parameters():
            param.data.clamp_(-weight_clip, weight_clip)

        # Reset gradient
        generator.zero_grad()
        discriminator.zero_grad()

        # Sample and score fake data
        fake_sample = generate(batch_size, NOISE_DIM, generator)
        discriminator_fake = discriminator(fake_sample)

        # Calculate generator loss
        # Generator wants discriminator to assign a high score to fake data
        generator_loss = -torch.mean(discriminator_fake)

        generator_loss.backward()
        generator_solver.step()

        # Reset gradient
        generator.zero_grad()
        discriminator.zero_grad()

        # Print training losses
        if int(iteration % epoch_length) == 0:
            epoch = int(iteration / epoch_length)
            logger.info('Epoch {}\n'
                         'Discriminator loss: {}; '
                         'Generator loss: {}'
                         .format(epoch,
                                 discriminator_loss.data.numpy(),
                                 generator_loss.data.numpy()))