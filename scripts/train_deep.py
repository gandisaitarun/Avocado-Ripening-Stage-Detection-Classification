import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader

from src.dataset import (
    DEFAULT_IMAGE_SIZE,
    AvocadoDataset,
    discover_images,
    stratified_train_validation_split,
)
from src.dl_model import build_classifier, save_checkpoint


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    if training:
        model.features.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += float(loss.item()) * labels.size(0)
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += labels.size(0)
    if total == 0:
        raise ValueError("A training or validation data loader is empty.")
    return total_loss / total, correct / total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a transfer-learning MobileNetV2 avocado classifier."
    )
    parser.add_argument("--train-dir", type=Path, default=Path("train"))
    parser.add_argument(
        "--model-path", type=Path, default=Path("models/mobilenet_v2.pth")
    )
    parser.add_argument(
        "--history-path",
        type=Path,
        default=Path("report/results/dl_history.csv"),
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        parser.error("--epochs, --patience, and --batch-size must be positive.")
    if args.image_size < 1 or args.learning_rate <= 0:
        parser.error("--image-size and --learning-rate must be positive.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths, labels, class_names = discover_images(args.train_dir)
    splits = stratified_train_validation_split(
        paths, labels, random_state=args.seed
    )
    train_paths, train_labels = splits["train"]
    val_paths, val_labels = splits["validation"]

    train_dataset = AvocadoDataset(
        train_paths,
        train_labels,
        class_names,
        training=True,
        image_size=args.image_size,
    )
    val_dataset = AvocadoDataset(
        val_paths,
        val_labels,
        class_names,
        training=False,
        image_size=args.image_size,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    model = build_classifier(
        len(class_names), pretrained=not args.no_pretrained
    ).to(device)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.asarray(class_names),
        y=np.asarray(train_labels),
    )
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
    )

    history = []
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_accuracy = run_epoch(model, val_loader, criterion, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": val_loss,
                "validation_accuracy": val_accuracy,
            }
        )
        print(
            f"Epoch {epoch:02d}/{args.epochs}: "
            f"train loss={train_loss:.4f}, train accuracy={train_accuracy:.4f}, "
            f"val loss={val_loss:.4f}, val accuracy={val_accuracy:.4f}"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping after epoch {epoch}.")
                break

    if best_state is None:
        raise RuntimeError("Training completed without a best validation checkpoint.")
    model.load_state_dict(best_state)
    save_checkpoint(model, class_names, args.model_path, args.image_size)
    args.history_path.parent.mkdir(parents=True, exist_ok=True)
    with args.history_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    print(f"Device: {device}")
    print(f"Saved best-validation model to {args.model_path}")
    print(f"Saved training history to {args.history_path}")


if __name__ == "__main__":
    main()
