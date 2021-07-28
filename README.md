# cDCGAN Quickdraw

PyTorch implementation of the Conditional Deep Convolutional Generative Adversarial Networks (DCGAN) for Google's Quick, Draw! [2] dataset.

## Introduction
Generative Adversarial Networks (GAN) are implicit generative deep learning models that capture the training data's distribution to be able to create new data from that same distribution. GANs were first introduced by Ian Goodfellow in 2014 [3]. As described in the paper, GANs are made up of two models: the *generator* and the *discriminator*. The *generator* can be thought of like a painting forger and the *discriminator* can be thought of as the detective. The forger (generator) is constantly trying to outsmart the detective (discriminator) by generating better and better fakes, while the detective is working to become better at correctly classifying the real and fake images. The DCGAN [1] extends this idea by using convolutional and convolutional-transpose layers in the discriminator and generator, respectively. 

## The c in cDCGAN
We can generate data conditioned on class labels by providing the GAN network with label data *y* [4]. Providing our network with label data will have two outcomes:
1. The discrimination will now have label data, which will help with training, making for a stronger discriminator.
2. The generator will now be able to draw images of the given label.

## References

[1] [https://arxiv.org/pdf/1511.06434.pdf](https://arxiv.org/pdf/1511.06434.pdf)

[2] [https://quickdraw.withgoogle.com/data](https://quickdraw.withgoogle.com/data)

[3] [http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf](http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)

[4] [https://arxiv.org/pdf/1411.1784.pdf](https://arxiv.org/pdf/1411.1784.pdf)
