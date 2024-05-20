"""This module contain functions for working with model."""
import torch
import cv2

from ultralytics import YOLO
from pathlib import Path

PROJECT_ROOT = Path().resolve().parents[0]

NAME_MODEL = "trained_yolov8n.pt"
PARAMS_EVAL = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    # "imgsz": 640,
    # "iou": 0.7,
    "conf": 0.20,
}

MODEL = YOLO(PROJECT_ROOT / "models" / NAME_MODEL)


def evaluate_model(path_file: Path) -> bool:
    """Evaluate model on images."""
    result = MODEL(path_file, save=False, **PARAMS_EVAL)

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
            results = MODEL(frame, save=False, **PARAMS_EVAL)

            annotated_frame = results[0].plot()
            cv2.imshow("Forest Fire Detection", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if cv2.getWindowProperty("Forest Fire Detection", cv2.WND_PROP_VISIBLE) < 1:
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


def open_web_camera_with_model():
    """Open WebCam with model evaluating."""
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Web Camera")

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            results = MODEL(frame, save=False, **PARAMS_EVAL)

            annotated_frame = results[0].plot()
            cv2.imshow("Web Camera", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if cv2.getWindowProperty("Web Camera", cv2.WND_PROP_VISIBLE) < 1:
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
