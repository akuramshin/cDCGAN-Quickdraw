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
import torchvision.utils as vutils
from architecture import Generator, Discriminator
import numpy as np
import matplotlib.pyplot as plt
import itertools
from torchvision import datasets, transforms
import matplotlib.animation as animation
from IPython.display import HTML

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
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
num_epochs = 10

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1


# We can use an image folder dataset the way we have it setup.
# Create the dataset
# dataset = QuickdrawDataset(datapath="data/X.npy",
#                            targetpath="data/y.npy",
#                            transform=transforms.Compose([
#                                transforms.Resize((28,28)),
#                                transforms.ToTensor(),
#                                transforms.Normalize((0.15,), (0.3038,)), # Mean and std of the dataset
#                            ]))

# # Create the dataloader
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

# MNIST Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.1307, ), std=(0.3081, ))
])

train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transform, download=True)

# Data Loader
dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda:0" if (cudnn.is_available() and ngpu > 0) else "cpu")

# Values come from paper
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)



netG = Generator().to(device)
netG.apply(weights_init)

netD = Discriminator().to(device)
netD.apply(weights_init)

criterion = nn.BCELoss()

fixed_noise = torch.randn(16, nz, 1, 1, device=device)
#fixed_noise = torch.randn(64, 1, 1, nz, device=device)
fixed_label = torch.nn.functional.one_hot(torch.Tensor([[3]*64]).long(), 10).view(64,10,1,1).to(device)
real_label = 0.9
fake_label = 0.

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

#std_list = [0.1-(0.1*i/(1600*5)) for i in range(1600*5)]
print("Starting training loop...")
for epoch in range(num_epochs):
        for i, (real_images, _) in enumerate(dataloader):
            bs = real_images.shape[0]
            ##############################
            #   Training discriminator   #
            ##############################

            netD.zero_grad()
            #std = std_list[iters]
            #instance_noise = (torch.randn(data[0].size(0), 1, 28, 28) * std).to(device)
            real_images = real_images.to(device)
            label = torch.full((bs,), real_label, dtype=torch.float, device=device)

            output = netD(real_images, 1)
            label = label.to(torch.float32)
            lossD_real = criterion(output, label)
            lossD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(bs, nz, 1, 1, device=device)
            fake_images = netG(noise, 1)
            #instance_noise = (torch.randn(b_size, 1, 28, 28) * std).to(device)
            label.fill_(fake_label)
            output = netD(fake_images.detach(), 1)
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
            output = netD(fake_images, 1)
            lossG = criterion(output, label)
            lossG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            G_losses.append(lossG.item())
            D_losses.append(lossD.item())

            if (i+1)%100 == 0:
                print('Epoch [{}/{}], step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, Discriminator - D(G(x)): {:.2f}, Generator - D(G(x)): {:.2f}'.format(epoch+1, num_epochs, 
                                                            i+1, len(dataloader), lossD.item(), lossG.item(), D_x, D_G_z1, D_G_z2))

        # Check how the generator is doing by saving G's output on fixed_noise
        netG.eval()
        generated_fake_images = netG(fixed_noise, 1)




# for epoch in range(num_epochs):
#     for i, data in enumerate(dataloader, 0):
#         # Update D network: maximize log(D(x)) + log(1 - D(G(z)))
#         # 1.) Train with real images
#         netD.zero_grad()
#         #std = std_list[iters]

#         #instance_noise = (torch.randn(data[0].size(0), 1, 28, 28) * std).to(device)
#         real = data[0].to(device)
#         #real_cpu += instance_noise
#         b_size = real.size(0)
#         label_fill = fill[data[1]].to(device)

#         label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
#         #label_real = torch.full((b_size,), real_label, dtype=torch.float, device=device)

#         output = netD(real, label_fill)
#         errD_real = criterion(output, label)
#         errD_real.backward()
#         D_x = output.mean().item()

#         # 2.) Train with fake
#         #z_noise = torch.randn(b_size, 1, 1, nz, device=device)
#         z_noise = torch.randn(b_size, nz, 1, 1, device=device)

#         y_noise = (torch.rand(b_size, 1)*10).type(torch.LongTensor).squeeze()
#         y = onehot[y_noise].to(device)
#         y_fill = fill[y_noise].to(device)

#         fake = netG(z_noise, y)
#         #instance_noise = (torch.randn(b_size, 1, 28, 28) * std).to(device)
#         output = netD(fake.detach(), y_fill)

#         label.fill_(fake_label)
#         errD_fake = criterion(output, label)
#         errD_fake.backward()
#         D_G_z1 = output.mean().item()
#         errD = errD_real + errD_fake

#         optimizerD.step()

#         # Update G network: maximize log(D(G(z)))
#         netG.zero_grad()
#         label.fill_(real_label)

#         # z_noise = torch.randn(b_size, nz, 1, 1, device=device)
#         # y_noise = (torch.rand(b_size, 1)*10).type(torch.LongTensor).squeeze()
#         # y = onehot[y_noise].to(device)
#         # y_fill = fill[y_noise].to(device)

#         #fake = netG(z_noise, y)
#         output = netD(fake.detach(), y_fill)

#         errG = criterion(output, label)
#         errG.backward()
#         D_G_z2 = output.mean().item()
#         optimizerG.step()

#         # Output training stats
#         if i % 50 == 0:
#             print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
#                   % (epoch, num_epochs, i, len(dataloader),
#                      errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

#         # Save Losses for plotting later
#         G_losses.append(errG.item())
#         D_losses.append(errD.item())
#         #std_change.append(std)

#         # Check how the generator is doing by saving G's output on fixed_noise
#         if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
#             with torch.no_grad():
#                 fake = netG(fixed_noise, fixed_label).detach().cpu()
#             img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

#         iters += 1

plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.plot(std_change,label="σ")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig('images/plot4.png')
#plt.show()

# fig = plt.figure(figsize=(8,8))
# plt.axis("off")
# ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
# ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

# HTML(ani.to_jshtml())

# Grab a batch of real images from the dataloader
real_batch = next(iter(dataloader))

# Plot the real images
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))
plt.imsave('images/real.png', np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)).numpy(), cmap="gray")

# Plot the fake images from the last epoch
# plt.subplot(1,2,2)
# plt.axis("off")
# plt.title("Fake Images")
# plt.imshow(np.transpose(img_list[-1],(1,2,0)))
# plt.imsave('images/fake2.png', np.transpose(img_list[-1],(1,2,0)).numpy(), cmap="gray")
# plt.show()
fig, ax = plt.subplots(4, 4, figsize=(6,6))
for i, j in itertools.product(range(4), range(4)):
    ax[i,j].get_xaxis().set_visible(False)
    ax[i,j].get_yaxis().set_visible(False)
for k in range(16):
    i = k//4
    j = k%4
    ax[i,j].cla()
    ax[i,j].imshow(generated_fake_images[k].data.cpu().numpy().reshape(28,28), cmap='Greys')
label = 'Epoch_{}'.format(epoch+1)
fig.text(0.5, 0.04, label, ha='center')
fig.suptitle('Fixed Noise')
fig.savefig("images/Epoch_{}.png".format(num_epochs))
