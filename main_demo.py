import argparse
from pathlib import Path

import torch
from PIL import Image

from src.classical_model import load_classical_model, predict_classical
from src.dataset import image_transforms
from src.dl_model import load_classifier, predict_deep


def run_demo(image_path: Path, classical_path: Path, deep_path: Path) -> None:
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    classical_model, classical_classes = load_classical_model(classical_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    deep_model, deep_classes, image_size = load_classifier(deep_path, device)
    if list(classical_classes) != list(deep_classes):
        raise ValueError("The saved models use different class label orders.")

    classical_label, classical_confidence = predict_classical(
        classical_model, image_path
    )
    with Image.open(image_path) as image:
        tensor = image_transforms(False, image_size)(image.convert("RGB"))
    deep_index, deep_confidence = predict_deep(deep_model, tensor, device)
    print(f"Image: {image_path}")
    print(
        f"Classical HSV/CIELAB/LBP + SVM: {classical_label} "
        f"({classical_confidence:.1%})"
    )
    print(
        f"MobileNetV2: {deep_classes[deep_index]} ({deep_confidence:.1%})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict an avocado ripening stage with both trained models."
    )
    parser.add_argument("image", type=Path, help="Path to an avocado image")
    parser.add_argument(
        "--classical-model", type=Path, default=Path("models/classical_svm.joblib")
    )
    parser.add_argument(
        "--deep-model", type=Path, default=Path("models/mobilenet_v2.pth")
    )
    args = parser.parse_args()
    run_demo(args.image, args.classical_model, args.deep_model)


if __name__ == "__main__":
    main()
