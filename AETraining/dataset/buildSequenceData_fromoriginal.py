import json
import os
import random
import numpy as np
from tqdm import tqdm

import torch
from tqdm import tqdm
import random
import seaborn as sns
from pylab import rcParams

from SkeletonData.data import *
from SkeletonData.array_segment import *

sns.set(style='whitegrid', palette='muted', font_scale=1.2)

HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]

sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

rcParams['figure.figsize'] = 12, 8

import gc
gc.collect()

with open("E:\\FYP_Data\\NTU120\\shapes_keys.json","r") as f0:
    id2shapes = json.load(f0)

root_dir = "D:\\FYP\\HAR-ZSL-XAI\\AETraining"
main_dir = "D:\\FYP\\HAR-ZSL-XAI"
data_dir = os.path.join("E:\\FYP_Data\\NTU120\skel\\nturgbd_skeletons_s001_to_s032\\nturgb+d_skeletons")
remove_files = ["E:\\FYP_Data\\NTU120\\skel\\NTU_RGBD120_samples_with_missing_skeletons.txt","E:\\FYP_Data\\NTU120\\skel\\NTU_RGBD_samples_with_missing_skeletons.txt"]
refined_data = os.path.join(main_dir,"data","sequence_data","midpoint_50f")
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 1-train_ratio - val_ratio
batch_size = 32

class_names = list(range(120))

files_to_remove = set()
for __f in remove_files:
    with open(__f,"r") as f0:
        for val in f0.read().split("\n"):
            files_to_remove.add(val)

len(files_to_remove)

total_files = set([x.split(".")[0] for x in os.listdir(data_dir)]) - files_to_remove
total_files_loc = set([f"{os.path.join(data_dir,x)}.skeleton" for x in total_files])

len(total_files)

builder = SkeletonFileBuilder(file_names=total_files_loc)
file_iterator = iter(builder)

from SkeletonData.array_segment import *
from SkeletonData.visualize import *

from functools import partial
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

os.makedirs(refined_data,exist_ok=True)
#for each_file in tqdm(files,desc="Files Used",total=len(files)):
with ThreadPoolExecutor() as executor:
    file_loc = list(tqdm(executor.map(partial(split_array_from_builder,refined_data,id2shapes),file_iterator), total=len(file_iterator),desc="Processed Files:"))

