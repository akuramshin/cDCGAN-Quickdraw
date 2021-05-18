from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from dataset import QuickdrawDataset
from noise_transform import AddGaussianNoise
import torchvision.transforms as transforms
import torchvision.utils as vutils

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Root directory for dataset
datapath = "data/data.h5"

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 128

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 28

# Number of channels in the training images. For color images this is 3
nc = 1

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 1

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# Number of categories (labels)
ncat = 10

# We can use an image folder dataset the way we have it setup.
# Create the dataset
dataset = QuickdrawDataset(datapath="data/X.npy",
                           targetpath="data/y.npy",
                           transform=transforms.Compose([
                               transforms.Resize((28,28)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.15,), (0.3038,)), # Mean and std of the dataset
                           ]))

# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
device = torch.device("cuda:0" if (cudnn.is_available() and ngpu > 0) else "cpu")
#device = "cpu"
print(device)

# Values come from paper
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)




class Generator(torch.nn.Module):
    """
    This is our generator model ("the artist") that learns to
    create images that look real.
    """

    def __init__(self):
        super(Generator, self).__init__()

        self.y_deconv = nn.Sequential(
            nn.ConvTranspose2d(10, ngf*4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
        )

        self.z_deconv = nn.Sequential(
            nn.ConvTranspose2d(100, ngf*4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
        )

        self.main = torch.nn.Sequential(
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*2, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, z, y):
        y = self.y_deconv(y.float())
        z = self.z_deconv(z)
        inp = torch.cat([z, y], 1)
        outp = self.main(inp)

        return outp


class Discriminator(nn.Module):
    """
    This is our discriminator model ("the detective") that learns to
    differentiate between real and fake images.
    """

    def __init__(self):
        super(Discriminator, self).__init__()

        self.y_conv = nn.Sequential(
            nn.Conv2d(10, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.x_conv = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.main = nn.Sequential(
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*4, ndf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        y = self.y_conv(y)
        x = self.x_conv(x)
        inp = torch.cat([x,y], 1)
        outp = self.main(inp)

        return outp.view(-1, 1).squeeze(1)

netG = Generator().to(device)
netG.apply(weights_init)

netD = Discriminator().to(device)
netD.apply(weights_init)

criterion = nn.BCELoss()

fixed_noise = torch.randn(64, nz, 1, 1, device=device)
fixed_label = torch.nn.functional.one_hot(torch.Tensor([[3]*64]).long(), 10).view(64,10,1,1).to(device)
real_label = 1.
fake_label = 0.

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

onehot = torch.zeros(10, 10)
onehot = onehot.scatter_(1, torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).view(10,1), 1).view(10, 10, 1, 1)
fill = torch.zeros([10, 10, image_size, image_size])
for i in range(10):
    fill[i, i, :, :] = 1

std = 0.1
print("Starting training loop...")
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        # Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        # 1.) Train with real images
        netD.zero_grad()

        real_cpu = (data[0] + (torch.randn(128, 1, 28, 28) * std)).to(device)
        b_size = real_cpu.size(0)
        y_fill = fill[torch.argmax(data[1], dim=1)].to(device)

        label_real = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        label_fake = torch.full((b_size,), fake_label, dtype=torch.float, device=device)

        output = netD(real_cpu, y_fill).view(-1)
        errD_real = criterion(output, label_real)
        errD_real.backward()
        D_x = output.mean().item()

        # 2.) Train with fake
        z_noise = torch.randn(b_size, nz, 1, 1, device=device)
        y_noise = (torch.rand(b_size, 1)*10).type(torch.LongTensor).squeeze()
        y_label = onehot[y_noise].to(device)
        y_fill = fill[y_noise].to(device)

        fake = netG(z_noise, y_label)
        instance_noise = (torch.randn(128, 1, 28, 28) * std).to(device)
        output = netD(fake.detach() + instance_noise, y_fill).view(-1)

        errD_fake = criterion(output, label_fake)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake

        # Update D
        #errD.backward()
        optimizerD.step()

        # Update G network: maximize log(D(G(z)))
        netG.zero_grad()

        z_noise = torch.randn(b_size, nz, 1, 1, device=device)
        y_noise = (torch.rand(b_size, 1)*10).type(torch.LongTensor).squeeze()
        y_label = onehot[y_noise].to(device)
        y_fill = fill[y_noise].to(device)

        fake = netG(z_noise, y_label)
        output = netD(fake, y_fill).view(-1)

        errG = criterion(output, label_real)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise, fixed_label).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1

plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig('plot2.png')
#plt.show()

fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

HTML(ani.to_jshtml())

# Grab a batch of real images from the dataloader
real_batch = next(iter(dataloader))

# Plot the real images
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
#plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))
plt.imsave('real.png', np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)).numpy(), cmap="gray")

# Plot the fake images from the last epoch
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.imsave('fake2.png', np.transpose(img_list[-1],(1,2,0)).numpy(), cmap="gray")
plt.show()
