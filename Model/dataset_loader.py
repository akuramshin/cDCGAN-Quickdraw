import numpy as np
from sklearn.model_selection import train_test_split
from os import walk, remove
import h5py
import requests


def download_data(categories):
    for cat in categories:
        cat_name = cat.replace(" ", "%20")
        url = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{}.npy".format(cat_name)
        r = requests.get(url, allow_redirects=True)
        open("temp/{}.npy".format(cat), 'wb').write(r.content)


def combine_data():
    mypath = "temp/"
    filename_list = []
    for (dirpath, dirnames, filenames) in walk(mypath):
        filename_list.extend(filenames)
        break

    X = []
    Y = []
    i = 0
    for filename in filename_list:
        path = mypath + filename
        x = np.load(path)
        x = x.astype('float32') / 255.0
        y = [i] * len(x)
        if i > 0:
            X = np.concatenate((x, X), axis=0)
            Y = np.concatenate((y, Y), axis=0)
        else:
            X = x
            Y = y
        i += 1
        remove(path)
    
    with h5py.File('data/X_full.h5', 'w') as hf:
        hf.create_dataset("name-of-dataset",  data=X)
    with h5py.File('data/Y_full.h5', 'w') as hf:
        hf.create_dataset("name-of-dataset",  data=Y)


download_data(["onion", "octopus", "nose"])
combine_data()