from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
import torchvision
from torchvision import transforms, utils
if __name__ == '__main__':
    from fastai.vision import *
    from fastai.widgets import *

metadata = pd.read_json("yoga-poses-dataset-master/pose-list-with-meta.json")
#print(metadata)
pose_pic_name = []

for name in metadata['sanskrit_name']:
    # modifying pose name to match format of image files
    name = name.join(name.split()).lower()
    name = name.replace(' ', '')
    pose_pic_name.append(name)

#print(pose_pic_name)

# add the images for each pose as another column in metadata


BASE_PATH = './dataset_test'
EXAMPLE_PATH = './example.jpg'

# Set the file path
path = Path(BASE_PATH)

# Create data loader / manager
data_bunch = ImageDataBunch.from_folder(
    # Data director
    path,
    # Reserve 20 percent of our images for our validation set
    valid_pct=0.2,
    # Transforms to apply to the image to create variations on our training image
    ds_tfms=get_transforms(max_zoom=1.0),
    # Dimension of image to process
    size=224,
    # Num workers to use
    num_workers=4
    #batch size (set for each set?
).normalize(
    # Use imagenet stats to normalize (to match what was pre-trained with)
    imagenet_stats
)

