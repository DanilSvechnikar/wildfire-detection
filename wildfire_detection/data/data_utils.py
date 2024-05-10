"""This module contains data utils functions."""
from tqdm import tqdm
from pathlib import Path

import pandas as pd
import shutil
import cv2


def resize_and_save_img(
    stream_data: tqdm,
    size_image: tuple[int, int],
    save_path: Path,
) -> None:
    """Resize and save images."""
    for fpath in stream_data:
        image = cv2.imread(str(fpath))
        resized_image = cv2.resize(image, size_image)
        cv2.imwrite(str(save_path / fpath.name), resized_image)


def modify_fpath(fpath: Path) -> str:
    """Modify path string."""
    trim_len = len(fpath.name) - len(fpath.suffix)
    return fpath.name[:trim_len]


def copy_splitting(
    df: pd.DataFrame,
    img_fpath: Path,
    labels_fpath: Path,
    save_dir: Path,
) -> None:
    """Copy splitting files in directories."""
    for row in tqdm(df.iterrows()):
        img_fname, fname = row[1]
        label_fname = fname + ".txt"

        img_file = img_fpath / img_fname
        label_file = labels_fpath / label_fname

        shutil.copy(img_file, save_dir / "images")
        shutil.copy(label_file, save_dir / "labels")
