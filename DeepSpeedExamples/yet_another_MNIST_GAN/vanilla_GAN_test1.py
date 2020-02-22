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
    sys.path.append(os.getcwd())
    from dcgan.dc_utils import Logger
    from dcgan.random_dataset import RandomDataset

    criterion = nn.BCELoss()

    batch_size = 4
    num_epochs = 20
    # Load data
    data = mnist_data()
    # Create loader with data, so that we can iterate over it
    mnist_data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    # Num batches
    num_batches = len(mnist_data_loader)

    generator_random_dataset = RandomDataset(batch_size=batch_size,
                                             sample_size=100)

    discriminator = DiscriminatorNet()
    discriminator_parameters = filter(lambda p: p.requires_grad, discriminator.parameters())

    generator = GeneratorNet()
    generator_parameters = filter(lambda p: p.requires_grad, generator.parameters())

    args = add_argument()

    if torch.cuda.is_available():
        discriminator.cuda()
        generator.cuda()

    # okay. fuck it. let's just give it a crack, shall we?

    discriminator_model_engine, discriminator_optimizer, discriminator_train_loader, __ = \
        deepspeed.initialize(args=args,
                             model=discriminator,
                             model_parameters=discriminator_parameters,
                             training_data=mnist_data_loader)

    generator_model_engine, generator_optimizer, generator_trainloader, __ = \
        deepspeed.initialize(args=args,
                             model=generator,
                             model_parameters=generator_parameters,
                             # training data is a function that provides
                             # just random data - generator, perhaps?
                             training_data=generator_random_dataset)

    # Create targets for the discriminator network D
    # (can use label flipping or label smoothing)
    real_labels = Variable(torch.ones(batch_size, 1))
    fake_labels = Variable(torch.zeros(batch_size, 1))

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(discriminator_train_loader):

            # TODO: 1) TRAIN DISCRIMINATOR
            # TODO: Evaluate the discriminator on the real input images
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(discriminator_model_engine.local_rank), \
                             data[1].to(discriminator_model_engine.local_rank)
            # feeding the discriminator with real data
            outputs = discriminator_model_engine(inputs)
            real_score = outputs
            # Compute the discriminator loss with respect to the real labels (1s)
            d_loss_real = criterion(outputs, real_labels)

            # TODO: what about the data[0].to(discriminator_model_engine.local_rank) thing?
            # TODO: do we have to do that?!

            # MAKE SOME NOIZE!!!
            batch_number, z = enumerate(generator_random_dataset)
            # Transform the noise through the generator network to get synthetic images
            fake_images = generator_model_engine(z)
            # Evaluate the discriminator on the fake images
            outputs = discriminator_model_engine(fake_images)
            fake_score = outputs
            # Compute the discriminator loss with respect to the fake labels (0s)
            d_loss_fake = criterion(outputs, fake_labels)

            # Optimize the discriminator
            d_loss = d_loss_real + d_loss_fake

            discriminator_model_engine.backward(d_loss)
            discriminator_model_engine.step()

            # 2) TRAIN GENERATOR
            # Draw random noise vectors as inputs to the generator network
            # Transform the noise through the generator network to get synthetic images
            batch_no, z = enumerate(generator_random_dataset)
            # Transform the noise through the generator network to get synthetic images
            fake_images = generator_model_engine(z)
            # Evaluate the (new) discriminator on the fake images
            outputs = discriminator_model_engine(fake_images)
            # Compute the cross-entropy loss with "real" as target (1s). This is what the G wants to do
            g_loss = criterion(outputs, real_labels)
            # Backprop it ...
            generator_model_engine.backward(g_loss)
            generator_model_engine.step()

            if (batch_number + 1) % 300 == 0:
                print('Epoch [%d/%d],  d_loss: %.4f, '
                      'g_loss: %.4f, Mean D(x): %.2f, Mean D(G(z)): %.2f'
                      % (epoch,
                         num_epochs,
                         d_loss.data[0],
                         g_loss.data[0],
                         real_score.data.mean(),
                         fake_score.data.mean())
                      )


if __name__ == "__main__":
    _main_()