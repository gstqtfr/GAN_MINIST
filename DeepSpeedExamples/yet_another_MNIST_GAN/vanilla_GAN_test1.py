import torch
from torch import nn, optim
from torch.utils.data import Dataset, ConcatDataset
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
import deepspeed
import argparse

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

    args=parser.parse_args()

    return args

def mnist_data(DATA_FOLDER='./data'):
    compose = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[0.5], std=[0.5])
         #transforms.Normalize((.5, .5, .5), (.5, .5, .5))
        ])
    out_dir = '{}/dataset'.format(DATA_FOLDER)
    return datasets.MNIST(root=out_dir,
                          train=True,
                          transform=compose,
                          download=True)


class NoiseDataset(Dataset):

    def __init__(self, n, m):
        super(Dataset, self).__init__()
        self.n = n
        self.m = m

    #def __getitem__(self, idx):
    #    if torch.is_tensor(idx):
    #        idx = idx.tolist()


    def __len__(self):
        return self.m

    def __add__(self, other):
        return ConcatDataset([self, other])

    def __iter__(self):
        fresh_batch = Variable(torch.randn(self.n, self.m))
        for r in fresh_batch:
            yield r

class DiscriminatorNet(torch.nn.Module):
    """
    A three hidden-layer discriminative neural network
    """

    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        n_features = 784
        n_out = 1

        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.out = nn.Sequential(
            torch.nn.Linear(256, n_out),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x

def images_to_vectors(images):
    return images.view(images.size(0), 784)

def vectors_to_images(vectors):
    return vectors.view(vectors.size(0), 1, 28, 28)

class RandomNoiseDataset():
    pass


class GeneratorNet(torch.nn.Module):
    """
    A three hidden-layer generative neural network
    """

    def __init__(self):
        super(GeneratorNet, self).__init__()
        n_features = 100
        n_out = 784

        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.LeakyReLU(0.2)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2)
        )

        self.out = nn.Sequential(
            nn.Linear(1024, n_out),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x

# Noise
def noise(size):
    n = Variable(torch.randn(size, 100))
    if torch.cuda.is_available(): return n.cuda()
    return n


def real_data_target(size):
    '''
    Tensor containing ones, with shape = size
    '''
    data = Variable(torch.ones(size, 1))
    if torch.cuda.is_available(): return data.cuda()
    return data

def fake_data_target(size):
    '''
    Tensor containing zeros, with shape = size
    '''
    data = Variable(torch.zeros(size, 1))
    if torch.cuda.is_available():
        return data.cuda()
    return data


def train_discriminator(discriminator, loss, optimizer, real_data, fake_data):
    # Reset gradients
    optimizer.zero_grad()

    # 1.1 Train on Real Data
    prediction_real = discriminator(real_data)
    # Calculate error and backpropagate
    error_real = loss(prediction_real, real_data_target(real_data.size(0)))
    error_real.backward()

    # 1.2 Train on Fake Data
    prediction_fake = discriminator(fake_data)
    # Calculate error and backpropagate
    error_fake = loss(prediction_fake, fake_data_target(real_data.size(0)))
    error_fake.backward()

    # 1.3 Update weights with gradients
    optimizer.step()

    # Return error
    return error_real + error_fake, prediction_real, prediction_fake


def train_generator(discriminator, loss, optimizer, fake_data):
    # 2. Train Generator
    # Reset gradients
    optimizer.zero_grad()
    # Sample noise and generate fake data
    prediction = discriminator(fake_data)
    # Calculate error and backpropagate
    error = loss(prediction, real_data_target(prediction.size(0)))
    error.backward()
    # Update weights with gradients
    optimizer.step()
    # Return error
    return error

def _main_():
    import os, sys
    sys.path.insert(0, os.getcwd())
    from dcgan.dc_utils import Logger

    batch_size = 4
    # Load data
    data = mnist_data()
    # Create loader with data, so that we can iterate over it
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    # Num batches
    num_batches = len(data_loader)

    discriminator = DiscriminatorNet()
    discriminator_parameters = filter(lambda p: p.requires_grad, discriminator.parameters())


    generator = GeneratorNet()
    generator_parameters = filter(lambda p: p.requires_grad, generator.parameters())

    args = add_argument()

    if torch.cuda.is_available():
        discriminator.cuda()
        generator.cuda()

    # okay. fuck it. let's just give it a crack, shall we?

    model_engine, optimizer, trainloader, __ = \
        deepspeed.initialize(args=args,
                             model=discriminator,
                             model_parameters=discriminator_parameters,
                             training_data=data_loader)

    model_engine, optimizer, trainloader, __ = \
        deepspeed.initialize(args=args,
                             model=generator,
                             model_parameters=generator_parameters,
                             # training data is a function that provides
                             # just random data - generator, perhaps?
                             training_data=data_loader)



    # Optimizers
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)

    # Loss function
    loss = nn.BCELoss()

    # Number of steps to apply to the discriminator
    d_steps = 1  # In Goodfellow et. al 2014 this variable is assigned to 1
    # Number of epochs
    num_epochs = 20

    num_test_samples = 16
    test_noise = noise(num_test_samples)

    logger = Logger(model_name='VGAN', data_name='MNIST')

    for epoch in range(num_epochs):
        for n_batch, (real_batch, _) in enumerate(data_loader):

            # 1. Train Discriminator
            real_data = Variable(images_to_vectors(real_batch))
            if torch.cuda.is_available(): real_data = real_data.cuda()
            # Generate fake data
            fake_data = generator(noise(real_data.size(0))).detach()
            # Train D
            d_error, d_pred_real, d_pred_fake = train_discriminator(
                discriminator,
                loss,
                d_optimizer,
                real_data,
                fake_data
            )

            # 2. Train Generator
            # Generate fake data
            fake_data = generator(noise(real_batch.size(0)))
            # Train G
            g_error = train_generator(
                discriminator,
                loss,
                g_optimizer,
                fake_data
            )
            # Log error
            logger.log(d_error, g_error, epoch, n_batch, num_batches)

            # Display Progress
            if (n_batch) % 100 == 0:
                # display.clear_output(True)
                # Display Images
                test_images = vectors_to_images(generator(test_noise)).data.cpu()
                logger.log_images(test_images, num_test_samples, epoch, n_batch, num_batches);
                # Display status Logs
                logger.display_status(
                    epoch, num_epochs, n_batch, num_batches,
                    d_error, g_error, d_pred_real, d_pred_fake
                )
            # Model Checkpoints
            logger.save_models(generator, discriminator, epoch)

if __name__ == "__main__":
    _main_()