# YOLO12 Animal Detection Training

This folder contains the training logic for the YOLO12 animal detection model.

## Structure

- `train_yolo12.ipynb` - Jupyter notebook with data augmentation, training, and evaluation
- `test_kinect_yolo12.py` - Test script for running inference with Kinect camera
- `README.md` - This file

## Setup

### 1. Install Dependencies

First, install all required libraries:

```bash
pip install -r ../requirements.txt
```

Or install individually:
```bash
pip install ultralytics albumentations matplotlib pandas seaborn pillow torch torchvision jupyter ipykernel pyyaml opencv-python numpy
```

### 2. Prepare Dataset

The dataset should be in the `../animales/` folder with the following structure:
```
animales/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

The `data.yaml` file should contain:
- `train`: path to training images
- `val`: path to validation images
- `test`: path to test images
- `nc`: number of classes
- `names`: list of class names

## Training

### Step 1: Open the Jupyter Notebook

```bash
jupyter notebook train_yolo12.ipynb
```

Or use Jupyter Lab:
```bash
jupyter lab train_yolo12.ipynb
```

### Step 2: Run the Notebook

The notebook includes:
1. **Data Augmentation Setup** - Visualize dataset and configure augmentation
2. **Model Training** - Train YOLO12 model with data augmentation
3. **Model Evaluation** - Evaluate on validation and test sets
4. **Model Export** - Export model to different formats (ONNX, TorchScript, etc.)

### Training Configuration

You can modify the training parameters in the notebook:
- `epochs`: Number of training epochs (default: 100)
- `batch`: Batch size (default: 16, adjust based on GPU memory)
- `imgsz`: Image size (default: 640)
- `lr0`: Initial learning rate (default: 0.01)
- Augmentation parameters (mosaic, flip, HSV, etc.)

### Model Variants

You can choose different YOLO12 model sizes:
- `yolo12n.pt` - Nano (fastest, smallest)
- `yolo12s.pt` - Small
- `yolo12m.pt` - Medium
- `yolo12l.pt` - Large
- `yolo12x.pt` - Extra Large (slowest, most accurate)

Change the model initialization in the notebook:
```python
model = YOLO('yolo12n.pt')  # Change to yolo12s, yolo12m, etc.
```

## Testing with Kinect Camera

### Prerequisites

1. Ensure OpenNI2 is installed and Kinect is connected
2. Train the model first using the notebook
3. The trained model should be at:
   - `../yolo_training/runs/yolo12_animal_detection/weights/best.pt`
   - Or `../yolo_training/exports/best.pt`

### Run the Test Script

```bash
python test_kinect_yolo12.py
```

### Controls

- Press `q` to quit the application

### Configuration

You can modify the following in `test_kinect_yolo12.py`:
- `CONFIDENCE_THRESHOLD`: Detection confidence threshold (default: 0.25)
- `MODEL_PATH`: Path to the trained model
- `OPENNI2_PATH`: Path to OpenNI2 installation

## Model Outputs

After training, you'll find:
- **Best model**: `runs/yolo12_animal_detection/weights/best.pt`
- **Last checkpoint**: `runs/yolo12_animal_detection/weights/last.pt`
- **Training plots**: `runs/yolo12_animal_detection/`
- **Exported models**: `exports/` (ONNX, TorchScript, etc.)

## Troubleshooting

### PyTorch Compatibility Issues (AttributeError: module 'torch' has no attribute '_utils')

**This is the most common issue!** If you see this error, your PyTorch version is too old.

**Solution:**
1. Run the upgrade script:
   ```bash
   python yolo_training/upgrade_pytorch.py
   ```

2. Or manually upgrade:
   ```bash
   pip install --upgrade torch torchvision torchaudio
   ```

3. **IMPORTANT:** After upgrading, restart your Jupyter kernel:
   - Go to `Kernel -> Restart Kernel` in Jupyter
   - Re-run all cells from the beginning

4. The notebook will automatically:
   - Use SGD optimizer instead of AdamW (more compatible)
   - Disable AMP (Automatic Mixed Precision)
   - Use YAML architecture if pretrained weights fail

**Minimum PyTorch version:** 2.1.0 or higher

### OpenNI2 Issues

If you get OpenNI2 initialization errors:
1. Ensure OpenNI2 is installed at `C:/Program Files/OpenNI2/Redist`
2. Close all Kinect applications (Kinect Studio, etc.)
3. Disconnect and reconnect the Kinect USB cable
4. Wait 5 seconds after closing applications

### Model Not Found

If the test script can't find the model:
1. Ensure you've completed training in the notebook
2. Check that `best.pt` exists in the weights folder
3. Update `MODEL_PATH` in `test_kinect_yolo12.py` if needed

### CUDA/GPU Issues

If you want to use CPU instead of GPU:
- The notebook will automatically detect and use CPU if CUDA is not available
- You can force CPU by setting `device = 'cpu'` in the notebook

### Memory Issues

If you run out of memory during training:
- Reduce `batch` size (e.g., from 16 to 8 or 4)
- Reduce `imgsz` (e.g., from 640 to 416)
- Use a smaller model variant (e.g., `yolo12n` instead of `yolo12m`)

## Notes

- Training time depends on your hardware (GPU recommended)
- The model will automatically save checkpoints during training
- You can resume training by setting `resume=True` in the training config
- Data augmentation is built into Ultralytics YOLO training


