# BDC Satria Data 2026 - Waste Classification

Complete solution for the Big Data Challenge Satria Data 2026 waste classification competition.

## Problem

Classify waste images into 3 categories:
- **Recyclable** (0): Non-electronic waste with recycling potential (plastic bottles, cans, paper, cardboard, glass)
- **Electronic** (1): Electronic waste/e-waste (phones, laptops, keyboards, chargers, cables)
- **Organic** (2): Biodegradable materials (leaves, fruits, vegetables, food waste, twigs)

## Project Structure

```
BDC2026/
├── data/
│   ├── train/
│   │   ├── Recyclable/
│   │   ├── Electronic/
│   │   └── Organic/
│   └── test/
├── model/
│   ├── best_model.pth
│   └── last_model.pth
├── output/
│   └── submission.csv
├── docs/
│   ├── training_history.png
│   └── confusion_matrix.png
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   ├── evaluate.py
│   └── predict.py
├── main.py
├── requirements.txt
└── README.md
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download the dataset from: https://bit.ly/datasetbdc2026
3. Extract the dataset to the `data/` directory

## Usage

### Basic Training

Run the complete pipeline with default settings:
```bash
python main.py
```

### Custom Training

Train with specific parameters:
```bash
python main.py \
    --data_dir data \
    --template_path output/submission.csv \
    --output_path output/submission.csv \
    --model_dir model \
    --docs_dir docs \
    --backbone efficientnet_b3 \
    --epochs 30 \
    --batch_size 64 \
    --learning_rate 1e-4 \
    --val_split 0.15
```

### Available Backbones

- `efficientnet_b0` (default) - Lightweight, fast training
- `efficientnet_b3` - Better accuracy, slower training
- `resnet50` - Classic architecture, good balance
- `resnet101` - Deeper network, better accuracy
- `convnext_tiny` - Modern architecture, excellent performance

### Arguments

- `--data_dir`: Directory containing train/test folders (relative or absolute, default: data)
- `--template_path`: Path to submission template file (relative or absolute, default: output/submission.csv)
- `--model_dir`: Directory to save trained models (relative or absolute, default: model)
- `--output_path`: Path to save submission file (relative or absolute, default: output/submission.csv)
- `--docs_dir`: Directory to save documentation and plots (relative or absolute, default: docs)
- `--backbone`: Pre-trained backbone model (default: efficientnet_b0)
- `--pretrained`: Use pre-trained weights (default: True)
- `--epochs`: Number of training epochs (default: 20)
- `--batch_size`: Batch size (default: 32)
- `--learning_rate`: Learning rate (default: 1e-3)
- `--weight_decay`: Weight decay (default: 1e-4)
- `--val_split`: Validation split ratio (default: 0.2)
- `--num_workers`: Data loading workers (default: 4)
- `--seed`: Random seed (default: 42)

## Pipeline Steps

1. **Data Loading**: Load training and test images
2. **Data Splitting**: Split training data into train/validation sets
3. **Data Augmentation**: Apply augmentations (flip, rotate, blur, etc.)
4. **Model Training**: Train with transfer learning using pre-trained CNN
5. **Model Evaluation**: Evaluate on validation set with F1-score
6. **Prediction**: Generate predictions on test set
7. **Submission**: Create submission CSV file

## Key Features

- **Transfer Learning**: Uses pre-trained models for better accuracy
- **Data Augmentation**: Robust augmentation pipeline for generalization
- **Class Balancing**: Stratified split to maintain class distribution
- **Early Stopping**: Saves best model based on validation F1-score
- **Learning Rate Scheduling**: Cosine annealing for better convergence
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-score, Confusion Matrix

## Model Architecture

The model uses a pre-trained CNN backbone with a custom classifier head:
- Backbone: EfficientNet-B0 (or other selected backbone)
- Classifier: 2-layer MLP with dropout for regularization
- Output: 3-class softmax for Recyclable, Electronic, Organic

## Evaluation Metric

The competition uses **Macro-averaged F1-Score**:
- F1-score calculated separately for each class
- Average of all class F1-scores
- Ensures balanced performance across all classes

## Tips for Better Performance

1. **Use stronger backbones**: `efficientnet_b3` or `convnext_tiny` for better accuracy
2. **Increase epochs**: Train for 30-50 epochs for convergence
3. **Adjust batch size**: Use larger batch size if GPU memory allows
4. **Ensemble models**: Train multiple models and average predictions
5. **Tune learning rate**: Try different learning rates (1e-4 to 1e-3)
6. **Cross-validation**: Use k-fold cross-validation for robust evaluation

## Submission

The submission file will be saved as `output/submission.csv` with the format:
```csv
id,predicted
1,0
2,1
3,2
...
```

Rename to `submission_NamaTim.csv` before submitting to the competition.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended for training)
- 8GB+ RAM
- Dataset: 26,527 training images, 1,458 test images

## License

This project is created for the BDC Satria Data 2026 competition.
