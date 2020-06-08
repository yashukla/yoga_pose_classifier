from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils



metadata = pd.read_json("yoga-poses-dataset-master/pose-list-with-meta.json")
print(metadata)

for name in metadata['sanskrit_name']:
    #modifying pose name to match format of image files
    name = name.join(name.split()).lower()
    name = name.replace(' ', '')
    print(name)

    # add the images for each pose as another column in metadata




