from __future__ import print_function
#%matplotlib inline
import argparse
import os
import imageio
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from dataset import QuickdrawDataset, Rescale
from noise_transform import AddGaussianNoise
import torchvision.utils as vutils
from architecture import Generator, Discriminator
import numpy as np
import matplotlib.pyplot as plt
import itertools
from torchvision import transforms
from matplotlib.ticker import MaxNLocator

# Set random seed for reproducibility
#manualSeed = 999
manualSeed = random.randint(1, 10000) # use if you want new results
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 128

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 28

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Number of training epochs
num_epochs = 2

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# If we want to train or test the model
train = True

# If we need to download the dataset
download = False

device = torch.device("cuda:0" if (cudnn.is_available() and ngpu > 0) else "cpu")

# Values come from paper
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def smooth_positive_labels(y):
    return y - 0.3 + (torch.rand(y.shape) * 0.5).to(device)

def smooth_negative_labels(y):
    return y + (torch.rand(y.shape) * 0.3).to(device)

def noisy_labels(y, p_flip):
    n_select = int(p_flip * y.shape[0])
    flip_ix = np.random.choice([i for i in range(y.shape[0])], size=n_select)
    y[flip_ix] = 1 - y[flip_ix]
    return y

# fixed_noise = torch.randn(16, nz, 1, 1, device=device)
# fixed_label = torch.nn.functional.one_hot(torch.Tensor([[9]*16]).long(), 10).view(16,10,1,1).to(device)

temp_noise = torch.randn(10, 100)
fixed_noise = temp_noise
fixed_label_noise = torch.zeros(10, 1)
for i in range(9):
    fixed_noise = torch.cat([fixed_noise, temp_noise], 0)
    temp = torch.ones(10, 1) + i
    fixed_label_noise = torch.cat([fixed_label_noise, temp], 0)

fixed_noise = fixed_noise.view(-1, 100, 1, 1).to(device)
fixed_label = torch.zeros(100, 10)
fixed_label.scatter_(1, fixed_label_noise.type(torch.LongTensor), 1)
fixed_label = fixed_label.view(-1, 10, 1, 1).to(device)

def save_epoch_result(epoch, num_images):
    netG.eval()
    generated_fake_images = netG(fixed_noise, fixed_label)
    netG.train()

    fig, ax = plt.subplots(num_images, num_images, figsize=(6,6))
    for i, j in itertools.product(range(num_images), range(num_images)):
        ax[i,j].get_xaxis().set_visible(False)
        ax[i,j].get_yaxis().set_visible(False)
    for k in range(num_images*num_images):
        i = k//num_images
        j = k%num_images
        ax[i,j].cla()
        ax[i,j].imshow(generated_fake_images[k].data.cpu().numpy().reshape(28,28), cmap='Greys')

    fig.text(0.5, 0.04, "Epoch {}".format(epoch), ha='center')
    fig.suptitle('Fixed Noise')
    fig.savefig("images/Epoch_{}.png".format(epoch))


if train:
    # We can use an image folder dataset the way we have it setup.
    # Create the dataset
    dataset = QuickdrawDataset(datapath="data/X.npy",
                            targetpath="data/y.npy",
                            transform=transforms.Compose([
                                transforms.Resize((28,28)),
                                transforms.ToTensor(),
                                Rescale(),
                                #transforms.Normalize((0.1575,), (0.3113,)), # Mean and std of the dataset
                        #         transforms.Normalize((-0.6849,), (0.6227,))
                            ]))

    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)


    netG = Generator().to(device)
    netG.apply(weights_init)

    netD = Discriminator().to(device)
    netD.apply(weights_init)

    criterion = nn.BCELoss()

    real_label = 1
    fake_label = 0

    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    std_change = []
    iters = 0
    generated_fake_images = None

    onehot = torch.zeros(10, 10)
    onehot = onehot.scatter_(1, torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).view(10,1), 1).view(10, 10, 1, 1)
    fill = torch.zeros([10, 10, image_size, image_size])
    for i in range(10):
        fill[i, i, :, :] = 1

    print("Starting training loop...")
    for epoch in range(num_epochs):
            for i, (real_images, y_) in enumerate(dataloader):
                bs = real_images.shape[0]

                ##############################
                #   Training discriminator   #
                ##############################

                # 1.) Train with real
                netD.zero_grad()
                std = max(0, 0.1 - ((0.1*iters)/35000))
                instance_noise = (torch.randn(bs, 1, 28, 28) * std).to(device)
                real_images = real_images.to(device) + instance_noise
                label = torch.full((bs,), real_label, dtype=torch.float32, device=device)
                label_fill = fill[y_].to(device)

                output = netD(real_images, label_fill).squeeze()
                label = smooth_positive_labels(label)
                lossD_real = criterion(output, label)
                lossD_real.backward()
                D_x = output.mean().item()

                # 2.) Train with fake
                noise = torch.randn(bs, nz, 1, 1, device=device)
                instance_noise = (torch.randn(bs, 1, 28, 28) * std).to(device)
                y_noise = (torch.rand(bs, 1)*10).type(torch.LongTensor).squeeze()
                y = onehot[y_noise].to(device)
                y_fill = fill[y_noise].to(device)

                fake_images = netG(noise, y) + instance_noise
                label.fill_(fake_label)
                output = netD(fake_images.detach(), y_fill).squeeze()
                #label = smooth_negative_labels(label)
                label = noisy_labels(label, 0.05)
                lossD_fake = criterion(output, label)
                lossD_fake.backward()
                D_G_z1 = output.mean().item()
                lossD = lossD_real + lossD_fake
                optimizerD.step()

                ##########################
                #   Training generator   #
                ##########################

                netG.zero_grad()

                label.fill_(real_label)
                noise = torch.randn(bs, nz, 1, 1, device=device)
                instance_noise = (torch.randn(bs, 1, 28, 28) * std).to(device)
                y_noise = (torch.rand(bs, 1)*10).type(torch.LongTensor).squeeze()
                y = onehot[y_noise].to(device)
                y_fill = fill[y_noise].to(device)
                fake_images = netG(noise, y) + instance_noise

                output = netD(fake_images, y_fill).squeeze()
                label = smooth_positive_labels(label)
                lossG = criterion(output, label)
                lossG.backward()
                D_G_z2 = output.mean().item()
                optimizerG.step()

                if (i+1)%150 == 0:
                    print('Epoch [{}/{}], step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, Discriminator - D(G(x)): {:.2f}, Generator - D(G(x)): {:.2f}'.format(epoch+1, num_epochs, 
                                                                i+1, len(dataloader), lossD.item(), lossG.item(), D_x, D_G_z1, D_G_z2))

                iters += 1

            # Check how the generator is doing by saving G's output on fixed_noise
            save_epoch_result(epoch, 10)

            # Save our training error
            G_losses.append(lossG.item())
            D_losses.append(lossD.item())
            std_change.append(std)

    fig, ax1 = plt.subplots()
    plt.title("Generator and Discriminator Loss During Training")
    ax1.plot(G_losses,label="G", color='tab:red')
    ax1.plot(D_losses,label="D", color='tab:blue')
    ax1.set_xlabel("iterations")
    ax1.set_ylabel("Loss")
    ax1.legend(loc=0)
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Save the plot of training error and instance noise
    ax2 = ax1.twinx()
    ax2.plot(std_change,label="σ", color='tab:green')
    ax2.set_ylabel("std")
    ax2.legend(loc=0)
    plt.savefig('images/plot.png')

    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloader))

    # Plot the real images
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))
    plt.imsave('images/real.png', np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)).numpy(), cmap="gray")

    # Save the models
    torch.save(netG.state_dict(), 'results/generator_param.pkl')
    torch.save(netD.state_dict(), 'results/discriminator_param.pkl')

    images = []
    for e in range(num_epochs):
        img_name = "images/Epoch_{}.png".format(e)
        images.append(imageio.imread(img_name))
    imageio.mimsave('results/generation_animation.gif', images, fps=5)

else:
    netG = Generator()
    netG.load_state_dict(torch.load('results/generator_param.pkl', map_location=torch.device(device)))

    save_epoch_result(24, 10)



