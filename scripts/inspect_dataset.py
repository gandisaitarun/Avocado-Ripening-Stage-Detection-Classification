import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageStat

from src.dataset import discover_images


def inspect_dataset(data_dir: Path, output_dir: Path) -> None:
    paths, labels, class_names = discover_images(data_dir)
    records = []
    corrupt_images = []
    for path, label in zip(paths, labels):
        try:
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                width, height = image.size
                stats = ImageStat.Stat(rgb)
                brightness = float(np.mean(stats.mean))
                brightness_std = float(np.mean(stats.stddev))
                grayscale = np.asarray(rgb.convert("L"), dtype=np.uint8)
                records.append(
                    {
                        "path": str(path),
                        "class": label,
                        "width": width,
                        "height": height,
                        "mean_brightness": round(brightness, 3),
                        "brightness_std": round(brightness_std, 3),
                        "dark_pixel_fraction": round(
                            float(np.mean(grayscale < 25)), 5
                        ),
                        "bright_pixel_fraction": round(
                            float(np.mean(grayscale > 230)), 5
                        ),
                    }
                )
        except (OSError, ValueError):
            corrupt_images.append(str(path))

    if not records:
        raise ValueError(f"No readable images found in {data_dir}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "image_analysis.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    counts = {name: 0 for name in class_names}
    for record in records:
        counts[record["class"]] += 1
    print("Readable image counts:")
    for class_name, count in counts.items():
        print(f"  {class_name}: {count}")
    print(f"Readable total: {len(records)}")
    print(f"Unreadable images: {len(corrupt_images)}")
    if corrupt_images:
        for path in corrupt_images:
            print(f"  Unreadable: {path}")
    print(f"Per-image lighting and resolution details: {csv_path}")

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(counts.keys(), counts.values(), color="forestgreen")
    axes[0].set_title("Readable images by class")
    axes[0].set_xlabel("Class")
    axes[0].set_ylabel("Image count")
    axes[0].tick_params(axis="x", rotation=20)

    brightness_by_class = [
        [
            record["mean_brightness"]
            for record in records
            if record["class"] == class_name
        ]
        for class_name in class_names
    ]
    axes[1].boxplot(brightness_by_class)
    axes[1].set_xticks(range(1, len(class_names) + 1), labels=class_names)
    axes[1].set_title("Mean image brightness by stage")
    axes[1].set_xlabel("Ripening stage")
    axes[1].set_ylabel("Mean RGB intensity (0-255)")
    axes[1].tick_params(axis="x", rotation=20)
    figure.tight_layout()
    plot_path = output_dir / "dataset_overview.png"
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)
    print(f"Overview chart: {plot_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize class counts, image resolution, and lighting indicators."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("train"))
    parser.add_argument("--output-dir", type=Path, default=Path("report/results/eda"))
    args = parser.parse_args()
    inspect_dataset(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
