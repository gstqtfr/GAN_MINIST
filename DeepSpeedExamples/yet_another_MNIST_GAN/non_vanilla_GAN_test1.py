import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets

def mnist_data(DATA_FOLDER='./data'):
    compose = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[0.5], std=[0.5])
         #transforms.Normalize((.5, .5, .5), (.5, .5, .5))
        ])
    out_dir = '{}/dataset'.format(DATA_FOLDER)
    return datasets.MNIST(root=out_dir, train=True, transform=compose, download=True)

def images_to_vectors(images):
    return images.view(images.size(0), 784)

def vectors_to_images(vectors):
    return vectors.view(vectors.size(0), 1, 28, 28)

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

        # TODO: JKK: yeah, i know, i know, but it's easier to RY when
        # TODO: JKK: you'r etrying to get the damn thing working.
        # TODO: JKK: put this into a single func, boolean param ...

        def freeze_discriminator():
            for name, param in self.named_parameters():
                if name.startswith('disc_'):
                    param.requires_grad = False
            for name, param in self.named_parameters():
                if name.startswith('gen_'):
                    param.requires_grad = True

        def freeze_generator():
            for name, param in self.named_parameters():
                if name.startswith('disc_'):
                    param.requires_grad = True
            for name, param in self.named_parameters():
                if name.startswith('gen_'):
                    param.requires_grad = False

        def forward(self, real_data):
            # we train either the discriminator or the generator
            # depending on the modulus of the iterations

            self.iterations = self.iterations + 1

            # odd iteration, so ...
            if self.iterations % 1 == 0:
                # train the discriminator ...
                freeze_generator()
                # 1.1 Train on Real Data
                # TODO: how do we present our stuff to self?
                prediction_real = self(real_data)

            else:
                # train the generator