# Avocado Image Classification

An end-to-end image classification workflow comparing a classical
HSV/CIELAB/LBP feature pipeline with a transfer-learning MobileNetV2 classifier.
The supplied dataset folders define the labels: `blackspot`, `canker`, `fresh`,
and `grenning`. This dataset is therefore a four-class condition classification
task, not the four ripeness stages described in the original assignment brief.

## Dataset setup

The repository includes separate training and test sets, with one subfolder per
class:

```text
train/
├── blackspot/
├── canker/
├── fresh/
└── grenning/
test/
├── blackspot/
├── canker/
├── fresh/
└── grenning/
```

Use the supplied folders (`blackspot`, `canker`, `fresh`, and `grenning`) in
both splits. Folder names become model labels.

Supported image formats are JPG, PNG, BMP, TIFF, and WebP (case-insensitive).
Training reserves a stratified 15% validation subset from `train/`; `test/` is
kept separate and is used only for final evaluation. At least two images per
class are required in `train/` to create that validation split.

## Setup

Use Python 3.8 or newer. A virtual environment is recommended. Install CPU or
CUDA-enabled PyTorch according to the official PyTorch installation selector,
then install the project dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For GPU training, install the CUDA-compatible `torch` and `torchvision` builds
before the remaining requirements, following the PyTorch instructions for your
system.

## Run the workflow

Run commands from the project root:

```powershell
# Explore training-set class balance, image dimensions, and lighting indicators
python -m scripts.inspect_dataset

# Train the HSV + CIELAB + LBP feature baseline and SVM
python -m scripts.train_classical

# Train MobileNetV2; pretrained ImageNet weights are used by default
python -m scripts.train_deep --epochs 20 --patience 5

# Compare both saved models on the supplied test set
python -m scripts.evaluate_models

# Run both models on one image
python main_demo.py "test/fresh/f (1).png"

# Build the assignment Word report after EDA and evaluation outputs exist
python -m scripts.build_report
```

The pretrained MobileNetV2 weights may need to be downloaded on the first run.
Use `--no-pretrained` to train without downloading weights (usually less
effective on a small dataset). Training uses CUDA when available and otherwise
uses CPU. Adjust batch size if device memory is limited.

## Outputs

- `report/results/eda/`: training-set class/lighting overview chart and per-image CSV.
- `models/classical_svm.joblib`: fitted classical feature pipeline and class
  names.
- `models/mobilenet_v2.pth`: best-validation MobileNetV2 checkpoint.
- `report/results/dl_history.csv`: per-epoch training and validation metrics.
- `report/results/model_comparison.json`: accuracy, macro precision, macro
  recall, macro F1, and per-class report for both models.
- `report/results/*_confusion_matrix.png`: test-set confusion matrices.
- `report/Assignment2_Avocado_Report.docx`: generated report with measured
  metrics, EDA summary, confusion matrices, and assignment-question responses.

Model weights and result files are ignored by Git. Run analysis, training, and
evaluation before writing a results section; no benchmark metrics are claimed
until the scripts are run in an environment with the dependencies installed.

## Assignment documentation

The four notebooks in `notebooks/` cover EDA, the classical baseline,
MobileNetV2 training, and held-out evaluation/demo. Run notebooks from a Jupyter
environment or upload the project to Colab with the dataset and model files
available. The report and external-submission checklist are in
`report/Assignment2_Avocado_Report.md` and `report/Submission_Guide.md`.

The provided dataset labels (`blackspot`, `canker`, `fresh`, `grenning`) do not
match the ripeness stages in the original assignment brief. Results in the
report are explicitly scoped to the supplied labels. Publishing a GitHub
repository, enabling Colab link access, recording the walkthrough, and entering
your personal time allocation require your account and personal input.

## Implementation notes

- Image files are discovered recursively under each class directory and
  class names are sorted to keep label indices deterministic.
- A fixed-seed stratified validation split (42 by default) is made only from
  `train/`; all final test metrics use the provided `test/` images. Training
  augmentations apply mild rotation, translation, scale, horizontal flips, and
  color jitter.
- The classical baseline uses normalized HSV and CIELAB channel histograms,
  uniform Local Binary Patterns, a standardized feature vector, and a
  class-balanced RBF SVM.
- MobileNetV2 uses an ImageNet-pretrained frozen feature backbone and a
  replaceable classification head. Class-weighted cross-entropy, Adam, and
  early stopping on validation loss address imbalance and overfitting.
- Evaluation reports macro-averaged precision, recall, and F1 alongside
  accuracy because accuracy alone may hide poor minority-class performance.

## Interpreting results and limitations

Lighting and camera/background differences can become shortcuts for either
model; inspect the EDA chart and per-image CSV, then validate on images captured
under different conditions. The provided labels represent visual image
categories; do not interpret the outputs as avocado ripeness or internal
firmness. Similar-looking conditions, glare, occlusion, and differences between
capture environments can still cause errors. Confidence values are model
scores, not calibrated probabilities or guarantees of food quality.

The report starter at [report/Assignment2_Avocado_Report.md](report/Assignment2_Avocado_Report.md)
includes the required discussion questions. Fill in the measured results and
your own time allocation after running the workflow. A PDF, public GitHub
repository, Colab link, and video walkthrough are submission tasks that require
your dataset, account access, and recording; they are not fabricated here.
