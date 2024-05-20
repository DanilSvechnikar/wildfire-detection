"""This module contains data utils functions."""
from pandas import DataFrame
from tqdm import tqdm
from pathlib import Path
from exif import Image

import pandas as pd
import shutil
import cv2

PROJECT_ROOT = Path().resolve().parents[0]


def get_coords_location(img_fpath) -> tuple[float, float]:
    """Return coords of image location as (lat, lon) tuple."""
    img_org = Image(img_fpath)

    if check_exif(img_org):
        try:
            decimal_latitude = dms_coords_to_dd_coords(
                img_org.gps_latitude, img_org.gps_latitude_ref,
            )
            decimal_longitude = dms_coords_to_dd_coords(
                img_org.gps_longitude, img_org.gps_longitude_ref,
            )

            return decimal_latitude, decimal_longitude
        except AttributeError:
            pass

    df_coords = get_synthetic_coords()

    decimal_latitude = df_coords["latitude"].values[0]
    decimal_longitude = df_coords["longitude"].values[0]
    place = df_coords["place"].values[0]

    return decimal_latitude, decimal_longitude


def get_synthetic_coords() -> DataFrame:
    """Return random synthetic coordinates from dataframe."""
    csv_coords = PROJECT_ROOT / "data" / "map_data" / "coords.csv"
    df_coords = pd.read_csv(csv_coords)
    return df_coords.sample()


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


def check_exif(img: Image) -> bool:
    """Check if exif contains."""
    if img.has_exif:
        return True

    return False


def dms_coords_to_dd_coords(coords: list[float], coords_ref: str) -> float:
    """Format coords of image."""
    decimal_degrees = coords[0] + coords[1] / 60 + coords[2] / 3600
    if coords_ref == "S" or coords_ref == "W":
        decimal_degrees = -decimal_degrees

    return decimal_degrees
