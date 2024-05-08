"""This module contain functions for working with model."""

import torch
import cv2

from ultralytics import YOLO
from pathlib import Path

PROJECT_ROOT = Path().resolve().parents[0]
MODEL = YOLO(PROJECT_ROOT / "runs" / "detect" / "train" / "weights" / "best.pt")


def evaluate_model(path_file: Path) -> bool:
    """Evaluate model on images."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    result = MODEL.predict(path_file, save=False, device=device)

    save_path = PROJECT_ROOT / "data" / "predicted"
    for res in result:
        name_file = Path(res.path).name
        save_path /= name_file

        res.save(filename=save_path)

    return True


def evaluate_model_video(path_file: Path) -> None:
    """Evaluate model on video."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cap = cv2.VideoCapture(str(path_file))

    cv2.namedWindow("Forest Fire Detection", cv2.WINDOW_NORMAL)
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            results = MODEL(frame, save=False, device=device)

            annotated_frame = results[0].plot()
            cv2.imshow("Forest Fire Detection", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
