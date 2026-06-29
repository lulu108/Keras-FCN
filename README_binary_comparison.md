# Binary Segmentation Comparison

This document describes the foreground/background comparison between traditional image segmentation methods and FCN predictions.

The workflow does not train models. It only creates traditional binary predictions, converts existing 21-class FCN predictions to binary masks, evaluates binary metrics, and saves side-by-side visualizations.

## Inputs

Expected AutoDL paths:

- `data/processed/images/val/`
- `data/processed/masks_binary/val/`
- `results/pred_21class/fcn32s/`
- `results/pred_21class/fcn8s/`

Ground-truth binary masks may contain `255` ignore pixels. Evaluation ignores those pixels.

## 1. Generate Traditional Predictions

Run all traditional methods on the first 50 validation images:

```bash
python scripts/traditional_methods.py --method all --num-images 50
```

Run one method:

```bash
python scripts/traditional_methods.py --method canny --num-images 50
python scripts/traditional_methods.py --method meanshift --num-images 50
python scripts/traditional_methods.py --method slic --num-images 50
```

Outputs:

- `results/pred_binary/canny/`
- `results/pred_binary/meanshift/`
- `results/pred_binary/slic/`
- `results/metrics/traditional_runtime.csv`

Each output mask is a single-channel PNG with pixel values `0` and `1`.

## 2. Convert FCN Predictions To Binary

Convert both FCN models:

```bash
python scripts/convert_pred_to_binary.py --model-name all
```

Or convert one model:

```bash
python scripts/convert_pred_to_binary.py --model-name fcn32s
python scripts/convert_pred_to_binary.py --model-name fcn8s
```

Conversion rule:

- class `0`: background
- classes `1-20`: foreground

Outputs:

- `results/pred_binary/fcn32s/`
- `results/pred_binary/fcn8s/`

## 3. Evaluate Binary Segmentation

Evaluate all available prediction directories:

```bash
python scripts/evaluate_binary_segmentation.py --method all
```

Evaluate one method:

```bash
python scripts/evaluate_binary_segmentation.py --method fcn8s
```

Metrics:

- Pixel Accuracy
- Foreground IoU
- Dice
- Mean Inference Time, when `results/metrics/traditional_runtime.csv` contains matching runtime records

Output:

- `results/metrics/metrics_binary.csv`

The evaluator only scores images that exist in the selected prediction directory and have a matching GT mask.

## 4. Visualize Comparisons

Generate 20 side-by-side comparison images:

```bash
python scripts/visualize_binary_compare.py --num-images 20
```

Each visualization contains:

1. Original image
2. GT binary
3. Canny
4. Mean Shift
5. SLIC
6. FCN-32s binary
7. FCN-8s binary

Output:

- `results/visual_compare_binary/`

## Suggested End-to-End Command Sequence

```bash
python scripts/traditional_methods.py --method all --num-images 50
python scripts/convert_pred_to_binary.py --model-name all
python scripts/evaluate_binary_segmentation.py --method all
python scripts/visualize_binary_compare.py --num-images 20
```

## Git Notes

Do not commit generated data, checkpoints, TFRecords, model weights, prediction masks, metrics CSVs, or visualization images. Commit only:

- `scripts/traditional_methods.py`
- `scripts/convert_pred_to_binary.py`
- `scripts/evaluate_binary_segmentation.py`
- `scripts/visualize_binary_compare.py`
- `README_binary_comparison.md`
