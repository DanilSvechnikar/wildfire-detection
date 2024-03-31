"""Model Utils."""
import torch

from ultralytics import YOLO
from pathlib import Path

PROJECT_ROOT = Path().resolve().parents[0]


def evaluate_model(path_file: Path) -> bool:
    """Evaluate model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = YOLO(PROJECT_ROOT / "runs" / "detect" / "train" / "weights" / "best.pt")
    result = model.predict(path_file, save=False, device=device)

    save_path = PROJECT_ROOT / "data" / "predicted"
    for res in result:
        name_file = Path(res.path).name
        save_path /= name_file

        res.save(filename=save_path)

    return True
