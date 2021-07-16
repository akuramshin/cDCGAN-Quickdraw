import torch
from dataset import QuickdrawDataset
import torch.utils.data
import torchvision.transforms as transforms
from dataset_loader import download_data, combine_data
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np

download_data(["cat", "envelope", "eyeglasses", "mushroom", "star", "baseball bat", "t-shirt", "car", "fish", "snail"])
combine_data(100000)
