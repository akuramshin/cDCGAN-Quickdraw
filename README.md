# cDCGAN Quickdraw

PyTorch implementation of the Deep Convolutional Generative Adversarial Networks (DCGAN) [1] for Google's Quick, Draw! [2] dataset.

## Introduction
Generative Adversarial Networks (GAN) are implicit generative deep learning models that capture the training data's distribution to be able to create new data from that same distribution. GANs were first introduced by Ian Goodfellow in 2014 [3]. As described in the paper, GANs are made up of two models: the *generator* and the *discriminator*. The *generator* can be thought of like a painting forger and the *discriminator* can be thought of as the detective. The forger (generator) is constantly trying to outsmart the detective (discriminator) by generating better and better fakes, while the detective is working to become better at correctly classifying the real and fake images. The DCGAN extends this idea by using convolutional and convolutional-transpose layers in the discriminator and generator, respectively. 

## References

[1] Radford, Alec, Luke Metz, and Soumith Chintala. "Unsupervised representation learning with deep convolutional generative adversarial networks." arXiv preprint arXiv:1511.06434 (2015).

Link to paper: [https://arxiv.org/pdf/1511.06434.pdf](https://arxiv.org/pdf/1511.06434.pdf)

[2] [https://quickdraw.withgoogle.com/data](https://quickdraw.withgoogle.com/data)

[3] Goodfellow, Ian, et al. "Generative adversarial nets." Advances in neural information processing systems. 2014.

Link to paper: [http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf](http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)
