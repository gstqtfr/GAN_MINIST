import torch
from torch import nn
from torch.optim import Adam
from torch.autograd import Variable

from torchvision import transforms, datasets


def mnist_data(DATA_FOLDER='./data'):
    compose = transforms.Compose(
        [
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    out_dir = '{}/dataset'.format(DATA_FOLDER)
    return datasets.MNIST(root=out_dir, train=True, transform=compose, download=True)


class DiscriminativeNet(torch.nn.Module):

    def __init__(self):
        super(DiscriminativeNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=128, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=512, out_channels=1024, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.out = nn.Sequential(
            nn.Linear(1024 * 4 * 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # Flatten and apply sigmoid
        x = x.view(-1, 1024 * 4 * 4)
        x = self.out(x)
        return x


class GenerativeNet(torch.nn.Module):

    def __init__(self):
        super(GenerativeNet, self).__init__()

        self.linear = torch.nn.Linear(100, 1024 * 4 * 4)

        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=1024, out_channels=512, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=512, out_channels=256, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=256, out_channels=128, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=128, out_channels=1, kernel_size=4,
                stride=2, padding=1, bias=False
            )
        )
        self.out = torch.nn.Tanh()

    def forward(self, x):
        # Project and reshape
        x = self.linear(x)
        x = x.view(x.shape[0], 1024, 4, 4)
        # Convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # Apply Tanh
        return self.out(x)

# Noise
def noise(size):
    n = Variable(torch.randn(size, 100))
    if torch.cuda.is_available(): return n.cuda()
    return n

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('BatchNorm') != -1:
        m.weight.data.normal_(0.00, 0.02)

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
    if torch.cuda.is_available(): return data.cuda()
    return data


def train_discriminator(discriminator, optimizer, loss, real_data, fake_data):
    # Reset gradients
    optimizer.zero_grad()

    # 1. Train on Real Data
    prediction_real = discriminator(real_data)
    # Calculate error and backpropagate
    error_real = loss(prediction_real, real_data_target(real_data.size(0)))
    error_real.backward()

    # 2. Train on Fake Data
    prediction_fake = discriminator(fake_data)
    # Calculate error and backpropagate
    error_fake = loss(prediction_fake, fake_data_target(real_data.size(0)))
    error_fake.backward()

    # Update weights with gradients
    optimizer.step()

    return error_real + error_fake, prediction_real, prediction_fake
    return (0, 0, 0)


def train_generator(discriminator, optimizer, loss, fake_data):
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
    # TODO: go on, treat yourself - get some args ...
    #DATA_FOLDER = './data/'

    data = mnist_data()
    batch_size = 100
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    num_batches = len(data_loader)

    # Create Network instances and init weights
    generator = GenerativeNet()
    generator.apply(init_weights)

    discriminator = DiscriminativeNet()
    discriminator.apply(init_weights)

    # TODO: JKK: this needs to when we get DeepSpeed enabled
    # Enable cuda if available
    if torch.cuda.is_available():
        generator.cuda()
        discriminator.cuda()

    # TODO: JKK: command-line args
    # Optimizers
    d_optimizer = Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    g_optimizer = Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Loss function
    loss = nn.BCELoss()

    # Number of epochs
    num_epochs = 200

    num_test_samples = 16
    test_noise = noise(num_test_samples)

    logger = Logger(model_name='DCGAN', data_name='MNIST')

    for epoch in range(num_epochs):
        for n_batch, (real_batch, _) in enumerate(data_loader):

            # 1. Train Discriminator
            real_data = Variable(real_batch)
            if torch.cuda.is_available(): real_data = real_data.cuda()
            # Generate fake data
            fake_data = generator(noise(real_data.size(0))).detach()
            # Train D
            # def train_discriminator(discriminator, optimizer, loss, real_data, fake_data)
            d_error, d_pred_real, d_pred_fake = train_discriminator(discriminator,
                                                                    d_optimizer,
                                                                    loss,
                                                                    real_data,
                                                                    fake_data)

            # 2. Train Generator
            # Generate fake data
            fake_data = generator(noise(real_batch.size(0)))
            # Train G
            # def train_generator(discriminator, optimizer, loss, fake_data)
            g_error = train_generator(discriminator, g_optimizer, loss, fake_data)
            # Log error
            logger.log(d_error, g_error, epoch, n_batch, num_batches)

            # Display Progress
            if (n_batch) % 100 == 0:
                #display.clear_output(True)
                # Display Images
                test_images = generator(test_noise).data.cpu()
                #logger.log_images(test_images, num_test_samples, epoch, n_batch, num_batches);
                # Display status Logs
                logger.display_status(
                    epoch, num_epochs, n_batch, num_batches,
                    d_error, g_error, d_pred_real, d_pred_fake
                )
            # Model Checkpoints
            logger.save_models(generator, discriminator, epoch)

if __name__ == "__main__":
    import os, sys
    # sys.path.append(os.path.join(os.path.dirname(__file__)))
    sys.path.append(os.path.join(os.getcwd()))
    from dc_utils import Logger
    _main_()