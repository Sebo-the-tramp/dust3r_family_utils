# MIT License
# 
# Copyright (c) 2025 Sebastian Cavada
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
check_datasets.py

Utility script for loading and inspecting datasets using the dust3r and mast3r frameworks.
This scripts loads the whole dataset as in training, it takes less than the training so we can find
any broken files before training starts.
You need to choose which dataset to test, it is therefore useful to test the loading dataset, before a major training run.

running with:
python check_datasets.py | tee check_datasets.log
will print any major errors or files not found to the console and save the output to a log file.

"""



# Standard Library Imports
import os
import sys
import copy
from pathlib import Path
from collections import defaultdict

# Modify sys.path to include local modules
sys.path.append("/home/sebastian.cavada/scsv/thesis/mast3r_complete")
sys.path.append("/home/sebastian.cavada/scsv/thesis/mast3r_complete/dust3r")

# Third-Party Imports
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from PIL import Image

# Local Application/Library Imports
import mast3r.utils.path_to_dust3r  # noqa
from mast3r.datasets import ARKitScenes, Habitat512
from mast3r.model import AsymmetricMASt3R
from dust3r.datasets import get_data_loader  # noqa
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.geometry import geotrf, inv, normalize_pointcloud
import dust3r.datasets
import croco.utils.misc as misc  # noqa

from dust3r.utils.geometry import (geotrf, inv, normalize_pointcloud)

# dust3r/mast3r
from dust3r.inference import inference
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images, rgb
from dust3r.utils.device import to_numpy
from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.model import AsymmetricCroCo3DStereo

import torch.backends.cudnn as cudnn

# Assign ARKitScenes to dust3r.datasets
dust3r.datasets.ARKitScenes = ARKitScenes
dust3r.datasets.Habitat = Habitat512

# Define infinity constant
inf = float('inf')

import time

seed = 777 + misc.get_rank()
torch.manual_seed(seed)
np.random.seed(seed)
cudnn.benchmark = False

dataset = "57_000 @ Habitat(1_000_000, split='train', resolution=[(224,224)], aug_crop='auto', aug_monocular=0.005, n_corres=8192, nneg=0.5, ROOT='/l/users/sebastian.cavada/datasets/habitat_processed')"

batch_size = 256
num_workers = 16
test = False

data_loader = get_data_loader(dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            pin_mem=False,
                            shuffle=not (test),
                            drop_last=not (test))


for epoch in range(30):
    print("#" * 20)
    print(f"Epoch {epoch}")
    data_loader.dataset.set_epoch(epoch)
    print("done loading dataset")

    tot = len(data_loader)

    for idx, batch in enumerate(data_loader): 
        pass
        # optional debugging code
        # print(f"Batch {idx}/{tot}")
        # print(f"")
        # for x in batch:
        #     print(x['label'])
        # view1, view2 = batch
