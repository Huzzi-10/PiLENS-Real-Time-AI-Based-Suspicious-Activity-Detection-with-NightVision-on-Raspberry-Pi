# Model Training

The repository includes `Model_Training/Model_Training.ipynb` for training the suspicious activity classifier.

## Training Output

The trained activity model should be saved as:

```text
Models/action_model.pth
```

## Runtime Models

PiLENS expects these files by default:

```text
Models/yolo11n.pt
Models/action_model.pth
```

You can override the paths in `.env`:

```env
YOLO_MODEL_PATH=Models/yolo11n.pt
CLASSIFIER_MODEL_PATH=Models/action_model.pth
```

## Dataset

Dataset download instructions are available in:

```text
Datasets/Download_Dataset.txt
```
