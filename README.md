# VisionProject

## Evaluating ConvNeXt-Tiny for image classification on different datasets

This repository provides a training and evaluation framework for the ConvNeXt-Tiny architecture applied to three image classification benchmarks:

- FGVC-Aircraft
- Oxford Flowers-102
- Food-101

Each dataset is trained and evaluated independently using a transfer learning approach. The models leverage pretrained weights on ImageNet-1K.

## Datasets

| Dataset       | Classes | Samples | Description                   |
| ------------- | ------- | ------- | ----------------------------- |
| FGVC-Aircraft | 102     | 10,000  | Fine-grained aircraft types   |
| Flowers-102   | 102     | 8,189   | Colorful flower categories    |
| Food-101      | 101     | 101,000 | Real-world food image dataset |

## Requirements

To install dependencies:

```bash
pip install torch torchvision scikit-learn matplotlib tqdm numpy
```

For running notebooks:

```bash
pip install notebook
```

## Project Structure

| File                      | Description                              |
| ------------------------- | ---------------------------------------- |
| `convnext_aircraft.ipynb` | Training and evaluation on FGVC-Aircraft |
| `convnext_flower.ipynb`   | Training and evaluation on Flowers-102   |
| `convnext_food.ipynb`     | Training and evaluation on Food-101      |

## Running Instructions

### Training

1. Open the appropriate notebook for the dataset.
2. Ensure all dependencies are installed.
3. Execute all cells to begin training.
4. The model will save:
   - The best weights as `best_convnext_model.pth`
   - Intermediate training states as `checkpoint.pth`

Training includes:

- Advanced augmentations during training only
- Mixed precision training via PyTorch AMP
- Early stopping based on validation performance

### Evaluation

Evaluation is performed automatically after training. The notebook computes and prints:

- Top-1 Accuracy
- Precision (macro average)
- Recall (macro average)
- F1-Score (macro average)

Example output:

```
Test Loss: 1.2746   Test Acc: 0.8557

Top-1 Accuracy: 0.8557
Precision: 0.8607
Recall: 0.8557
F1-Score: 0.8555
```

## Model Details

- Architecture: ConvNeXt-Tiny (torchvision)
- Initialization: Pretrained on ImageNet-1K
- Modifications:
  - Final classifier layer adjusted to dataset class count
  - Dropout added before final layer
- Optimizer: AdamW
- Scheduler: CosineAnnealingWarmRestarts
- Loss Function: CrossEntropyLoss with label smoothing
- Augmentations: RandAugment, ColorJitter, RandomAffine, RandomErasing

## Notes

- Validation and test transformations use only resizing and normalization.
- Resume training from saved checkpoints using the `checkpoint.pth` file.
- Training uses early stopping to avoid overfitting.

## Results

| Dataset       | Top-1 Accuracy | Precision | Recall | F1-Score |
| ------------- | -------------- | --------- | ------ | -------- |
| FGVC-Aircraft | 0.8542         | 0.8587    | 0.8542 | 0.8538   |
| Flowers-102   | 0.9809         | 0.9818    | 0.9809 | 0.9808   |
| Food-101      | 0.8687         | 0.8718    | 0.8687 | 0.8691   |



## License

This project is intended for academic and research purposes only.
