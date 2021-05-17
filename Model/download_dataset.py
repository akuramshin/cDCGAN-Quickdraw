import torch
from dataset import QuickdrawDataset
import torch.utils.data
import torchvision.transforms as transforms
from dataset_loader import download_data, combine_data
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np

download_data(["cat", "nose", "drill", "duck", "teddy-bear", "potato", "foot", "fish", "banana", "snail"])
combine_data(200000)
