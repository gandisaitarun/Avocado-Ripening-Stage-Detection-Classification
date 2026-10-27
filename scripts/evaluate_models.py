import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

from src.classical_model import extract_feature_matrix, load_classical_model
from src.dataset import AvocadoDataset, discover_images
from src.dl_model import load_classifier


def calculate_metrics(y_true, y_pred, class_names):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=class_names, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=class_names,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
    }


def save_confusion_matrix(y_true, y_pred, class_names, output_path, title):
    matrix = confusion_matrix(y_true, y_pred, labels=class_names)
    figure, axis = plt.subplots(figsize=(7, 6))
    display = axis.imshow(matrix, interpolation="nearest", cmap="Greens")
    figure.colorbar(display, ax=axis)
    axis.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title=title,
    )
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate both saved classifiers on the held-out test split."
    )
    parser.add_argument("--test-dir", type=Path, default=Path("test"))
    parser.add_argument(
        "--classical-model", type=Path, default=Path("models/classical_svm.joblib")
    )
    parser.add_argument(
        "--deep-model", type=Path, default=Path("models/mobilenet_v2.pth")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("report/results"))
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive.")

    test_paths, test_labels, class_names = discover_images(args.test_dir)

    classical, classical_classes = load_classical_model(args.classical_model)
    if list(classical_classes) != class_names:
        raise ValueError("Classical model classes do not match the dataset folders.")
    classical_features = extract_feature_matrix(test_paths)
    classical_predictions = classical.predict(classical_features)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    deep_model, deep_classes, image_size = load_classifier(args.deep_model, device)
    if list(deep_classes) != class_names:
        raise ValueError("Deep model classes do not match the dataset folders.")
    test_dataset = AvocadoDataset(
        test_paths, test_labels, class_names, training=False, image_size=image_size
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    deep_predictions = []
    deep_model.eval()
    with torch.inference_mode():
        for images, _ in test_loader:
            logits = deep_model(images.to(device))
            deep_predictions.extend(
                class_names[index] for index in logits.argmax(dim=1).cpu().tolist()
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, predictions in (
        ("classical_svm", classical_predictions),
        ("mobilenet_v2", deep_predictions),
    ):
        metrics = calculate_metrics(test_labels, predictions, class_names)
        results[name] = metrics
        save_confusion_matrix(
            test_labels,
            predictions,
            class_names,
            args.output_dir / f"{name}_confusion_matrix.png",
            f"{name.replace('_', ' ').title()} confusion matrix",
        )
        print(
            f"{name}: accuracy={metrics['accuracy']:.4f}, "
            f"macro precision={metrics['macro_precision']:.4f}, "
            f"macro recall={metrics['macro_recall']:.4f}, "
            f"macro F1={metrics['macro_f1']:.4f}"
        )
    output_path = args.output_dir / "model_comparison.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "test_image_count": len(test_paths),
                "class_names": class_names,
                "metrics": results,
            },
            file,
            indent=2,
        )
    print(f"Detailed metrics saved to {output_path}")
    print(f"Test device for MobileNetV2: {device}")


if __name__ == "__main__":
    main()
