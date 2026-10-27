import argparse
from pathlib import Path

from src.classical_model import save_classical_model, train_classical_model
from src.dataset import discover_images, stratified_train_validation_split


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the HSV/CIELAB/LBP + SVM baseline."
    )
    parser.add_argument("--train-dir", type=Path, default=Path("train"))
    parser.add_argument(
        "--model-path", type=Path, default=Path("models/classical_svm.joblib")
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths, labels, class_names = discover_images(args.train_dir)
    splits = stratified_train_validation_split(
        paths, labels, random_state=args.seed
    )
    train_paths, train_labels = splits["train"]
    model = train_classical_model(train_paths, train_labels, random_state=args.seed)
    save_classical_model(model, class_names, args.model_path)
    print(f"Trained on {len(train_paths)} images across {len(class_names)} classes.")
    print(f"Saved classical model to {args.model_path}")


if __name__ == "__main__":
    main()
