import numpy as np
from os import walk, remove
import requests
from tqdm import tqdm


def download_data(categories):
    for i in tqdm(range(len(categories))):
        cat = categories[i]
        cat_name = cat.replace(" ", "%20")
        url = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{}.npy".format(cat_name)
        r = requests.get(url, allow_redirects=True)
        open("temp/{}.npy".format(cat), 'wb').write(r.content)


# Creates a dataset of size *dataset_size* with evenly sized categories
def combine_data(dataset_size):
    mypath = "temp/"
    filename_list = []
    for (_, _, filenames) in walk(mypath):
        filename_list.extend(filenames)
        break

    X = []
    Y = []
    i = 0
    cat_size = int(dataset_size / len(filename_list))
    for filename in filename_list:
        path = mypath + filename
        x = np.load(path)
        y = [i] * len(x)
        print("{}: {}".format(i, filename))

        np.random.shuffle(x)
        x = x[:cat_size]
        y = y[:cat_size]

        if i > 0:
            X = np.concatenate((x, X), axis=0)
            Y = np.concatenate((y, Y), axis=0)
        else:
            X = x
            Y = y
        i += 1
        remove(path)
    
    np.save('data/X.npy', X)
    np.save('data/y.npy', Y)