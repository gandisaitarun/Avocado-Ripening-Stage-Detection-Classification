from pathlib import Path
from typing import Sequence, Tuple

import torch
from torch import nn
from torchvision import models


def build_classifier(num_classes: int, pretrained: bool = True) -> nn.Module:
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2.")
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    for parameter in model.features.parameters():
        parameter.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Linear(in_features, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(128, num_classes),
    )
    return model


def save_checkpoint(
    model: nn.Module,
    class_names: Sequence[str],
    output_path: Path,
    image_size: int,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "class_names": list(class_names),
            "image_size": image_size,
            "architecture": "mobilenet_v2",
        },
        output_path,
    )


def load_classifier(
    checkpoint_path: Path, device: torch.device
) -> Tuple[nn.Module, Sequence[str], int]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint.get("architecture") != "mobilenet_v2":
        raise ValueError(f"Unsupported model architecture in {checkpoint_path}.")
    class_names = checkpoint.get("class_names")
    if not class_names:
        raise ValueError(f"Checkpoint has no class_names: {checkpoint_path}")
    model = build_classifier(len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, class_names, int(checkpoint["image_size"])


def predict_deep(model: nn.Module, image_tensor: torch.Tensor, device: torch.device):
    model.eval()
    with torch.inference_mode():
        logits = model(image_tensor.unsqueeze(0).to(device))
        probabilities = torch.softmax(logits, dim=1)[0]
        confidence, index = torch.max(probabilities, dim=0)
    return int(index.item()), float(confidence.item())
