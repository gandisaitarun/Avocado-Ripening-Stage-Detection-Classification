from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def discover_images(data_dir: Path) -> Tuple[List[Path], List[str], List[str]]:
    """Return image paths, labels, and sorted class names from class subfolders."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {data_dir}. "
            "Place one folder per class inside the selected data directory."
        )

    class_names = sorted(
        folder.name for folder in data_dir.iterdir() if folder.is_dir()
    )
    if len(class_names) < 2:
        raise ValueError(f"Expected at least two class folders in {data_dir}.")

    paths: List[Path] = []
    labels: List[str] = []
    for class_name in class_names:
        class_paths = sorted(
            path
            for path in (data_dir / class_name).rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not class_paths:
            raise ValueError(
                f"No supported images found in class folder {data_dir / class_name}."
            )
        paths.extend(class_paths)
        labels.extend([class_name] * len(class_paths))

    if not paths:
        raise ValueError(f"No supported images found in class folders under {data_dir}.")
    return paths, labels, class_names


def stratified_split(
    paths: Sequence[Path],
    labels: Sequence[str],
    random_state: int = 42,
) -> Dict[str, Tuple[List[Path], List[str]]]:
    """Split data into stratified 70/15/15 train, validation, and test sets."""
    if len(paths) != len(labels):
        raise ValueError("The number of image paths and labels must match.")

    counts: Dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    too_small = {name: count for name, count in counts.items() if count < 3}
    if too_small:
        details = ", ".join(f"{name}: {count}" for name, count in too_small.items())
        raise ValueError(
            "Stratified 70/15/15 splitting requires at least 3 images per class; "
            f"found {details}."
        )

    train_paths, remainder_paths, train_labels, remainder_labels = train_test_split(
        list(paths),
        list(labels),
        test_size=0.30,
        random_state=random_state,
        stratify=list(labels),
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        remainder_paths,
        remainder_labels,
        test_size=0.50,
        random_state=random_state,
        stratify=remainder_labels,
    )
    return {
        "train": (train_paths, train_labels),
        "validation": (val_paths, val_labels),
        "test": (test_paths, test_labels),
    }


def stratified_train_validation_split(
    paths: Sequence[Path],
    labels: Sequence[str],
    validation_fraction: float = 0.15,
    random_state: int = 42,
) -> Dict[str, Tuple[List[Path], List[str]]]:
    """Split a provided training set into stratified train and validation sets."""
    if len(paths) != len(labels):
        raise ValueError("The number of image paths and labels must match.")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")

    counts: Dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    too_small = {name: count for name, count in counts.items() if count < 2}
    if too_small:
        details = ", ".join(f"{name}: {count}" for name, count in too_small.items())
        raise ValueError(
            "A stratified train/validation split requires at least 2 images per "
            f"class; found {details}."
        )

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        list(paths),
        list(labels),
        test_size=validation_fraction,
        random_state=random_state,
        stratify=list(labels),
    )
    return {
        "train": (train_paths, train_labels),
        "validation": (val_paths, val_labels),
    }


def image_transforms(training: bool, image_size: int = DEFAULT_IMAGE_SIZE):
    operations = [transforms.Resize((image_size, image_size))]
    if training:
        operations.extend(
            [
                transforms.RandomAffine(
                    degrees=15, translate=(0.08, 0.08), scale=(0.9, 1.1)
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            ]
        )
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transforms.Compose(operations)


class AvocadoDataset(Dataset):
    def __init__(
        self,
        paths: Sequence[Path],
        labels: Sequence[str],
        class_names: Sequence[str],
        training: bool = False,
        image_size: int = DEFAULT_IMAGE_SIZE,
    ) -> None:
        if len(paths) != len(labels):
            raise ValueError("The number of image paths and labels must match.")
        self.paths = list(paths)
        self.labels = list(labels)
        self.class_to_index = {name: index for index, name in enumerate(class_names)}
        unknown = set(self.labels) - self.class_to_index.keys()
        if unknown:
            raise ValueError(f"Labels missing from class_names: {sorted(unknown)}")
        self.transform = image_transforms(training, image_size)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
        except (OSError, ValueError) as error:
            raise RuntimeError(f"Could not read image {path}: {error}") from error
        return self.transform(image), self.class_to_index[self.labels[index]]
