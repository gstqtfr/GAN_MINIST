import torch
from torch import nn
from torch.nn import Module
from torch.optim import Adam
from torch.autograd import Variable
import torchvision
from torchvision import transforms, datasets
import argparse
import deepspeed

def add_argument():

    parser=argparse.ArgumentParser(description='DCGAN MNIST on DeepSpeed')

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

class HazyGAN(torch.nn.Module):
    """The HazyGAN class is specialised to allow us to run on DeepSpeed. All the
    weights, biases, layers, u.s.w. of a generator & discriminator are contained
    in a single class, & so are available the ZeRO optimizer. We choose to freeze
    the weights in the training cycle when the discriminator or the generator are
    being trained."""

    def __init__(self):
        super(HazyGAN, self).__init__()

        self.iterations = 0

        self.disc_conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=128, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.disc_conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.disc_conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.disc_conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=512, out_channels=1024, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.disc_out = nn.Sequential(
            nn.Linear(1024 * 4 * 4, 1),
            nn.Sigmoid(),
        )

        # this is for the generator layers - n.b. we have to freeze these
        # layers when the discriminator is being trained (& vice versa, of course!)

        self.gen_linear = torch.nn.Linear(100, 1024 * 4 * 4)

        self.gen_conv1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=1024, out_channels=512, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.gen_conv2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=512, out_channels=256, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.gen_conv3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=256, out_channels=128, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.gen_conv4 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=128, out_channels=1, kernel_size=4,
                stride=2, padding=1, bias=False
            )
        )
        self.gen_out = torch.nn.Tanh()

        def deep_freeze(freeze_generator=True):
            if freeze_generator:
                # we go through the generator weights & freeze
                # the *flip* out of 'em ...
                # for param in model.parameters():
                #     param.requires_grad = False
                for name, param in self.named_parameters():
                    if name.startswith('disc_'):
                        param.requires_grad = False
            else:
                for name, param in self.named_parameters():
                    if name.startswith('gen_'):
                        param.requires_grad = False

            # forward() method
            # loss = model_engine(batch)

            # runs backpropagation
            # model_engine.backward(loss)

            # weight update
            # model_engine.step()

        def forward(self, real_data):
            criterion = nn.BCELoss()
            # we need to be able to choose between discriminator
            # & generator here, which we're going to do by using
            # a modulus on the number of iterations which we maintain
            # in the object itself ...

            self.iterations = self.iterations+1

            # odd iteration, so ...
            if self.iterations % 1 == 0:
                # we call forward on the generator ...
                # ... so we need to affect only the generator portion of the network ...
                # ... so we freeze the discriminator weights ...
                deep_freeze(False)
                # ... but do we make some noise while we're here?!
                # TODO: train the generator on the discriminator's response
                # should have batch_size for 1st tensor dim.
                gen_noise = noise(4)
                # present it to our generator, get the output
                fake_data = generator_forward(gen_noise)
                # now we present it to our discriminator
                disc_results_fake_out = discriminator_forward(fake_data)
                disc_fake_error = criterion(disc_results_fake_out, Variable(torch.ones([1,1])))
                disc_fake_error.backward()
                # return the generator loss!
            else:
                # we call forward on the discriminator
                # ... & so we freeze the generator's weights ...
                deep_freeze(True)
                # TODO: train the discriminator on real data
                disc_results_real_out = discriminator_forward(real_data)
                # all of these are real data, so they all have a target of 1
                disc_real_error = criterion(disc_results_real_out, Variable(torch.ones([1, 1])))
                disc_real_error.backward()  # compute/store gradients, but don't change params
                # return the discriminator loss!


        # TODO: JKK: how do we tell which of these guys to forward?!
        def discriminator_forward(self, x):
            # Convolutional layers
            x = self.disc_conv1(x)
            x = self.disc_conv2(x)
            x = self.disc_conv3(x)
            x = self.disc_conv4(x)
            # Flatten and apply sigmoid
            x = x.view(-1, 1024 * 4 * 4)
            x = self.disc_out(x)
            return x

        def generator_forward(self, x):
            # Project and reshape
            x = self.gen_linear(x)
            x = x.view(x.shape[0], 1024, 4, 4)
            # Convolutional layers
            x = self.gen_conv1(x)
            x = self.gen_conv2(x)
            x = self.gen_conv3(x)
            x = self.gen_conv4(x)
            # Apply Tanh
            return self.gen_out(x)

# Noise
def noise(size):
    n = Variable(torch.randn(size, 100))
    # TODO: JKK: check this!!!
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
    # TODO: JKK: check this!!!
    if torch.cuda.is_available(): return data.cuda()
    return data

def fake_data_target(size):
    '''
    Tensor containing zeros, with shape = size
    '''
    data = Variable(torch.zeros(size, 1))
    # TODO: JKK: check this!!!
    if torch.cuda.is_available(): return data.cuda()
    return data

