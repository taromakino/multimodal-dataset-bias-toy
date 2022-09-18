import glob
import numpy as np
import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils.file import *

DPATH = os.path.join(os.environ["DATA_DPATH"], "modelnet40_images_new_12x")
CLASS_NAMES = [
    "airplane", "bathtub", "bed", "bench", "bookshelf", "bottle", "bowl", "car", "chair", "cone", "cup", "curtain",
    "desk", "door", "dresser", "flower_pot", "glass_box", "guitar", "keyboard", "lamp", "laptop", "mantel", "monitor",
    "night_stand", "person", "piano", "plant", "radio", "range_hood", "sink", "sofa", "stairs", "stool", "table",
    "tent", "toilet", "tv_stand", "vase", "wardrobe", "xbox"]
N_VIEWS = 12
FRONT_VIEW_IDX = 0
REAR_VIEW_IDX = 6

class ModelNet40Dataset(Dataset):
    def __init__(self, stage, clip_transforms):
        self.series = load_file(os.path.join(DPATH, f"{stage}_series.pkl"))
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            clip_transforms])

    def __len__(self):
        return len(self.series)

    def __getitem__(self, idx):
        x_all_views = torch.load(self.series.index[idx])
        x0 = self.transform(x_all_views[FRONT_VIEW_IDX])
        x1 = self.transform(x_all_views[REAR_VIEW_IDX])
        y = self.series.iloc[idx]
        return x0, x1, y

def make_numpy_imgs():
    for stage in ["train", "test"]:
        for class_name in CLASS_NAMES:
            class_fpaths = sorted(glob.glob(os.path.join(DPATH, class_name, stage, "*.png")))
            n_objects = int(len(class_fpaths) / N_VIEWS)
            for object_idx in range(n_objects):
                model_fpaths = class_fpaths[object_idx * N_VIEWS:(object_idx + 1) * N_VIEWS]
                imgs = [np.array(Image.open(f).convert("RGB")) for f in model_fpaths]
                with open(class_fpaths[object_idx * N_VIEWS].rsplit(".", 2)[0] + ".npy", "wb") as save_file:
                    torch.save(imgs, save_file)

def to_series(df):
    df.index = df.fpath
    df.drop("fpath", axis=1, inplace=True)
    return df.y

def make_data_dfs(train_ratio):
    rng = np.random.RandomState(0) # Always use the same seed for train/val split
    train_df = {"fpath": [], "y": []}
    val_df = {"fpath": [], "y": []}
    test_df = {"fpath": [], "y": []}
    for class_idx, class_name in enumerate(CLASS_NAMES):
        trainval_fpaths = glob.glob(os.path.join(DPATH, class_name, "train", "*.npy"))
        test_fpaths = glob.glob(os.path.join(DPATH, class_name, "test", "*.npy"))
        rng.shuffle(trainval_fpaths)
        n_train = int(len(trainval_fpaths) * train_ratio)
        train_fpaths = trainval_fpaths[:n_train]
        val_fpaths = trainval_fpaths[n_train:]
        train_df["fpath"] += train_fpaths
        val_df["fpath"] += val_fpaths
        test_df["fpath"] += test_fpaths
        train_df["y"] += [class_idx] * len(train_fpaths)
        val_df["y"] += [class_idx] * len(val_fpaths)
        test_df["y"] += [class_idx] * len(test_fpaths)
    save_file(to_series(pd.DataFrame(train_df)), os.path.join(DPATH, "train_series.pkl"))
    save_file(to_series(pd.DataFrame(val_df)), os.path.join(DPATH, "val_series.pkl"))
    save_file(to_series(pd.DataFrame(test_df)), os.path.join(DPATH, "test_series.pkl"))

def make_data(batch_size, n_workers, clip_transforms):
    train_data = DataLoader(ModelNet40Dataset("train", clip_transforms), shuffle=True, batch_size=batch_size,
        num_workers=n_workers, pin_memory=True)
    val_data = DataLoader(ModelNet40Dataset("val", clip_transforms), batch_size=batch_size, num_workers=n_workers,
        pin_memory=True)
    test_data = DataLoader(ModelNet40Dataset("test", clip_transforms), batch_size=batch_size, num_workers=n_workers,
        pin_memory=True)
    return train_data, val_data, test_data