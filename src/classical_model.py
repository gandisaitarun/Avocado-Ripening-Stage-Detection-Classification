from pathlib import Path
from typing import Dict, Sequence, Tuple

import cv2
import joblib
import numpy as np
from PIL import Image
from skimage.feature import local_binary_pattern
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def extract_features(image_path: Path, image_size: int = 128) -> np.ndarray:
    """Extract normalized HSV, CIELAB, and LBP histograms from an image."""
    try:
        with Image.open(image_path) as image:
            rgb = np.asarray(image.convert("RGB").resize((image_size, image_size)))
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Could not read image {image_path}: {error}") from error

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    histograms = []
    for image, channel_ranges in (
        (hsv, ((0, 180), (0, 256), (0, 256))),
        (lab, ((0, 256), (0, 256), (0, 256))),
    ):
        for channel, value_range in enumerate(channel_ranges):
            histogram = cv2.calcHist(
                [image], [channel], None, [16], list(value_range)
            )
            histogram = cv2.normalize(histogram, None, norm_type=cv2.NORM_L1)
            histograms.append(histogram.flatten())

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_histogram, _ = np.histogram(
        lbp.ravel(), bins=np.arange(0, 11), range=(0, 10)
    )
    lbp_histogram = lbp_histogram.astype(np.float32)
    lbp_histogram /= max(float(lbp_histogram.sum()), 1.0)
    return np.concatenate([*histograms, lbp_histogram]).astype(np.float32)


def extract_feature_matrix(paths: Sequence[Path]) -> np.ndarray:
    if not paths:
        raise ValueError("Cannot extract features from an empty image list.")
    return np.vstack([extract_features(path) for path in paths])


def train_classical_model(
    paths: Sequence[Path], labels: Sequence[str], random_state: int = 42
):
    if len(paths) != len(labels) or not paths:
        raise ValueError("Training images and labels must be non-empty and aligned.")
    features = extract_feature_matrix(paths)
    model = make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            random_state=random_state,
        ),
    )
    model.fit(features, labels)
    return model


def save_classical_model(
    model, class_names: Sequence[str], output_path: Path
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "class_names": list(class_names)}, output_path
    )


def load_classical_model(model_path: Path) -> Tuple[object, Sequence[str]]:
    bundle: Dict = joblib.load(model_path)
    if not isinstance(bundle, dict) or "model" not in bundle or "class_names" not in bundle:
        raise ValueError(f"Invalid classical model bundle: {model_path}")
    return bundle["model"], bundle["class_names"]


def predict_classical(model, image_path: Path):
    features = extract_features(image_path).reshape(1, -1)
    probabilities = model.predict_proba(features)[0]
    classes = model.classes_
    best = int(np.argmax(probabilities))
    return str(classes[best]), float(probabilities[best])
