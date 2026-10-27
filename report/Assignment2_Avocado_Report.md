# Avocado Image Classification — Assignment Report

> Dataset, metrics, and training details below were measured from the supplied
> repository data. Replace the remaining submission and personal-experience
> placeholders with your own information.

> Dataset scope: the repository's folders are `blackspot`, `canker`, `fresh`,
> and `grenning`. These are the labels actually trained and evaluated; they do
> not match the ripeness-stage labels in the original assignment brief.

## Dataset understanding

- Dataset source: `train/` and `test/` folders supplied in this repository.
- Training images by class: blackspot 184, canker 179, fresh 281, grenning 347
  (991 total).
- Test images by class: blackspot 22, canker 22, fresh 33, grenning 22
  (99 total).
- Unreadable training images: 0.
- Validation: 149 images stratified from the training directory; 842 images
  remained for fitting.
- Class balance: grenning is the largest training class (about 35%); canker is
  the smallest (about 18%). Review `report/results/eda/dataset_overview.png`
  and `image_analysis.csv` for lighting and resolution indicators.
- Resolution is class-correlated: blackspot, canker, and grenning training
  images are 800x800; fresh images vary substantially. Median mean brightness
  also varies by class (approximately 49 grenning, 95 canker, 103 blackspot,
  155 fresh on an 8-bit scale), creating shortcut-learning risk.

## Methodology and model rationale

The classical baseline combines HSV and CIELAB color histograms with uniform
Local Binary Pattern texture histograms, then fits a class-balanced RBF SVM.
This is a compact, interpretable feature-based baseline. MobileNetV2 provides a
lightweight transfer-learning comparison; its frozen ImageNet feature backbone
and trained classification head are useful when the labeled dataset is modest.
Both methods use the same stratified split and test set.

## Evaluation

Both saved models were evaluated on the same held-out 99-image test set. The
run used CPU-only PyTorch; MobileNetV2 was trained for 10 epochs, and its best
validation accuracy was 100% (validation loss 0.0369). These validation results
are separate from the test metrics below.

| Model | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---:|---:|---:|---:|
| HSV/CIELAB/LBP + SVM | 96.97% | 97.00% | 96.59% | 96.57% |
| MobileNetV2 | 95.96% | 95.63% | 95.45% | 95.45% |

On this split, the classical baseline exceeded MobileNetV2 by 1.01 percentage
points in accuracy. Both models classified all `fresh` and `grenning` test
images correctly. The SVM's remaining errors were concentrated in `canker`
(recall 86.36%) and false positives for `blackspot` (precision 88.00%).
MobileNetV2 had `blackspot` recall of 86.36% and `canker` precision of 87.50%.
These results describe this 99-image test set only; they do not establish
performance on new cameras, conditions, or datasets. See the saved confusion
matrices and per-class JSON reports for detail.

## Mandatory assignment questions

### 1. In production, how does your model perform?

The best test accuracy measured here was 96.97% from the SVM; MobileNetV2
scored 95.96%. Inference latency was not benchmarked. These are results on a
small held-out split and do not establish production performance. Test the
deployed camera, lighting, and capture conditions before operational use. This
dataset predicts its four folder labels, not ripeness stage.

### 2. What are the drawbacks of your model?

- Image labels capture visible categories and cannot establish internal
  firmness.
- One surface view can miss features on the opposite side of the fruit.
- Similar visual categories, label noise, class imbalance, and differences
  between capture conditions can affect predictions. Confidence scores are not
  guaranteed to be calibrated.

### 3. If given more time, what would you improve?

- Collect and label a larger, cultivar-diverse dataset with balanced classes
  and capture-condition metadata.
- Test external-camera and lighting-domain data, review failure cases, and
  calibrate confidence before any operational use.
- Explore multi-view imaging and non-destructive firmness or spectral sensors
  to address limitations of RGB surface color.
- Benchmark and, if appropriate, optimize the selected model for the actual
  target hardware (for example ONNX Runtime or TensorRT), measuring both
  latency and accuracy after conversion.

### 4. Where did you spend the most time while solving your assignment?

No personal work-time log was collected for this run, so a truthful percentage
breakdown cannot be supplied here. Complete this section with your own
approximate time allocation for dataset inspection/cleaning, feature and
augmentation experiments, model training/debugging, and evaluation/reporting.

## Reproducibility and submission

- Random seed: 42 unless overridden.
- Commands and dependency setup: see [README.md](../README.md).
- GitHub repository: [add URL after publishing]
- Colab notebook: [add shareable URL after configuring access]
- Walkthrough recording: [add link after recording]
- Word report: `Assignment2_Avocado_Report.docx` (regenerate with
  `python -m scripts.build_report`)
- Submission steps and 30-minute agenda: [Submission_Guide.md](Submission_Guide.md)
