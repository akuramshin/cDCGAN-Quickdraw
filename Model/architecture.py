import torch
import torch.nn as nn


# Number of channels in the training images. For color images this is 3
nc = 1

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 32

# Size of feature maps in discriminator
ndf = 32

# Number of categories (labels)
ncat = 10


# class Generator(nn.Module):
#     """
#     This is our generator model ("the artist") that learns to
#     create images that look real.
#     """

#     def __init__(self):
#         super(Generator, self).__init__()

#         self.y_deconv = nn.Sequential(
#             nn.ConvTranspose2d(ncat, ngf*4, 4, 1, 0, bias=False),
#             nn.BatchNorm2d(ngf*4),
#             nn.ReLU(True),
#         )

#         self.z_deconv = nn.Sequential(
#             nn.ConvTranspose2d(nz, ngf*4, 4, 1, 0, bias=False),
#             nn.BatchNorm2d(ngf*4),
#             nn.ReLU(True),
#         )

#         # self.main = torch.nn.Sequential(
#         #     nn.ConvTranspose2d(ngf*8, ngf*4, 4, 1, 0, bias=False),
#         #     nn.BatchNorm2d(ngf*4),
#         #     nn.ReLU(True),
#         #     nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
#         #     nn.BatchNorm2d(ngf*2),
#         #     nn.ReLU(True),
#         #     nn.ConvTranspose2d(ngf*2, nc, 4, 2, 1, bias=False),
#         #     nn.Tanh()
#         # )
#         # self.main = torch.nn.Sequential(
#         #     # nn.ConvTranspose2d(nz + ncat, ngf*2, 7, 1, 0, bias=False),
#         #     nn.ConvTranspose2d(nz, ngf*2, 7, 1, 0, bias=False),
#         #     nn.BatchNorm2d(ngf*2),
#         #     nn.ReLU(True),
#         #     nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
#         #     nn.BatchNorm2d(ngf),
#         #     nn.ReLU(True),
#         #     nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
#         #     nn.Tanh()
#         # )
#         self.reshape = torch.nn.Sequential(
#             nn.Linear(100, 128*7*7),
#             nn.LeakyReLU(0.2),
#         )
#         self.main = torch.nn.Sequential(
#             # nn.ConvTranspose2d(nz + ncat, ngf*2, 7, 1, 0, bias=False),
#             nn.ConvTranspose2d(nz, ngf*2, 4, 1, 0, bias=False),
#             nn.BatchNorm2d(ngf*2),
#             nn.ReLU(True),

#             nn.ConvTranspose2d(ngf*2, ngf, 3, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf),
#             nn.ReLU(True),

#             nn.ConvTranspose2d(ngf, 32, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(32),
#             nn.ReLU(True),

#             nn.ConvTranspose2d(32, nc, 4, 2, 1, bias=False),
#             nn.Tanh()
#         )
    
#     def forward(self, z, y):
#         # y = self.y_deconv(y.float())
#         # z = self.z_deconv(z)
#         #inp = torch.cat([z, y], 1)
#         #z = self.reshape(z)
#         #z = z.view(-1, 128, 7, 7)
#         #print(z.shape)
#         outp = self.main(z)

#         return outp

class Generator(nn.Module):
    def __init__(self):
      super(Generator, self).__init__()
      self.network = nn.Sequential(
          nn.ConvTranspose2d(nz, ngf*4, 4, 1, 0, bias=False),
          nn.BatchNorm2d(ngf*4),
          nn.ReLU(True),
  
          nn.ConvTranspose2d(ngf*4, ngf*2, 3, 2, 1, bias=False),
          nn.BatchNorm2d(ngf*2),
          nn.ReLU(True),
  
          nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
          nn.BatchNorm2d(ngf),
          nn.ReLU(True),
  
          nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
          nn.Tanh()
      )
  
    def forward(self, input, y):
      output = self.network(input)
      return output

# class Discriminator(nn.Module):
#     """
#     This is our discriminator model ("the detective") that learns to
#     differentiate between real and fake images.
#     """

#     def __init__(self):
#         super(Discriminator, self).__init__()

#         self.y_conv = nn.Sequential(
#             nn.Conv2d(ncat, ndf, 4, 2, 1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#         )

#         self.x_conv = nn.Sequential(
#             nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#         )

#         # self.main = nn.Sequential(
#         #     nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
#         #     nn.BatchNorm2d(ndf*4),
#         #     nn.LeakyReLU(0.2, inplace=True),
#         #     nn.Conv2d(ndf*4, ndf*8, 4, 1, 0, bias=False),
#         #     nn.BatchNorm2d(ndf*8),
#         #     nn.LeakyReLU(0.2, inplace=True),
#         #     nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False),
#         #     nn.Sigmoid()
#         # )
#         self.main = nn.Sequential(
#             # nn.Conv2d(nc+ncat, ndf*2, 5, 2, 0, bias=False),
#             nn.Conv2d(nc, 32, 4, 2, 1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),

#             nn.Conv2d(32, ndf, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf),
#             nn.LeakyReLU(0.2, inplace=True),

#             nn.Conv2d(ndf, ndf*2, 3, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf*2),
#             nn.LeakyReLU(0.2, inplace=True),

#             nn.Conv2d(ndf*2, 1, 4, 1, 0, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, x, y):
#         # y = self.y_conv(y)
#         # x = self.x_conv(x)
#         #inp = torch.cat([x,y], 1)
#         outp = self.main(x)

#         return outp.view(-1, 1).squeeze(1)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.network = nn.Sequential(
                
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                
                nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                
                nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )
    def forward(self, input, y):
        output = self.network(input)
        return output.view(-1, 1).squeeze(1)