import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import pickle
from torch.utils import data as t_data
import torchvision.datasets as datasets
from torchvision import transforms


# %matplotlib inline

data_transforms = transforms.Compose([transforms.ToTensor()])

mnist_trainset = datasets.MNIST(root='./data',
                                train=True,
                                download=True,
                                transform=data_transforms)

batch_size=4

dataloader_mnist_train = t_data.DataLoader(mnist_trainset,
                                           batch_size=batch_size,
                                           shuffle=True
                                          )

def make_some_noise():
    return torch.rand(batch_size,100)


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

gen = generator(100,784)
parameters = filter(lambda p: p.requires_grad, gen.parameters())
args=add_argument()

# defining discriminator class

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

dis = discriminator(784,1)

d_steps = 1
g_steps = 1

criteriond1 = nn.BCELoss()
optimizerd1 = optim.SGD(dis.parameters(), lr=0.001, momentum=0.9)

criteriond2 = nn.BCELoss()
optimizerd2 = optim.SGD(gen.parameters(), lr=0.001, momentum=0.9)

printing_steps = 200

epochs = 4000

for epoch in range(epochs):

    if epoch % printing_steps == 0:
        print(epoch)

    # training discriminator
    for d_step in range(d_steps):
        dis.zero_grad()

        # training discriminator on real data
        for inp_real, _ in dataloader_mnist_train:
            inp_real_x = inp_real
            break

        inp_real_x = inp_real_x.reshape(4, 784)
        dis_real_out = dis(inp_real_x)
        dis_real_loss = criteriond1(dis_real_out, Variable(torch.ones(batch_size, 1)))
        dis_real_loss.backward()

        # training discriminator on data produced by generator
        inp_fake_x_gen = make_some_noise()
        dis_inp_fake_x = gen(inp_fake_x_gen).detach()  # output from generator is generated
        dis_fake_out = dis(dis_inp_fake_x)
        dis_fake_loss = criteriond1(dis_fake_out, Variable(torch.zeros(batch_size, 1)))
        dis_fake_loss.backward()

        optimizerd1.step()

    # training generator
    for g_step in range(g_steps):
        gen.zero_grad()

        # generating data for input for generator
        gen_inp = make_some_noise()

        gen_out = gen(gen_inp)
        dis_out_gen_training = dis(gen_out)
        gen_loss = criteriond2(dis_out_gen_training, Variable(torch.ones(batch_size, 1)))
        gen_loss.backward()

        optimizerd2.step()

