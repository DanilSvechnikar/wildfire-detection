"""Model Utils."""
import numpy as np
import torch
import cv2

from typing import List
from numpy import typing as npt

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
    cap = cv2.VideoCapture(str(path_file))

    cv2.namedWindow("Forest Fire Detection", cv2.WINDOW_NORMAL)
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            results = MODEL(frame)

            annotated_frame = results[0].plot()
            cv2.imshow("Forest Fire Detection", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


# def process_output_data(result: List) -> npt.NDArray:
#     """Add bounding box and label to original image."""
#     img = result.orig_img
#     boxes = result.boxes.xyxy.cpu().numpy().astype(np.int32)
#     name_cls = result.boxes.cls.cpu().numpy().astype(np.int32)
#     conf = result.boxes.conf.cpu().numpy().astype(np.int32)
#
#     for box, cls, conf in zip(boxes, name_cls, conf):
#         cv2.rectangle(
#             img,
#             (box[0], box[1]),
#             (box[2], box[3]),
#             color=(0, 255, 0),
#             thickness=2,
#         )
#
#         cv2.putText(
#             img,
#             f"{result.names[cls]}",
#             (box[0], box[1]),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             fontScale=1.0,
#             color=(0, 255, 0),
#             thickness=2,
#         )
#
#     return img
