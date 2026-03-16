# Potato Leaf Disease Classification with ResNet50V2 🚀

[
[
[

**Production-ready CNN model achieving high accuracy on potato leaf disease detection using transfer learning with ResNet50V2 + interactive Streamlit demo.**

## ✨ Features
- **Transfer Learning**: ResNet50V2 pre-trained on ImageNet (256x256 input)
- **Data Augmentation**: Rotation, zoom, flip, shear for robust training
- **Fine-tuning**: Strategic unfreezing of top layers with learning rate scheduling
- **Class Balancing**: Handles imbalanced potato disease dataset
- **Interactive Demo**: Streamlit app for real-time leaf disease prediction
- **Production Optimized**: GPU memory growth, early stopping, model checkpointing

## 📊 Results
```
Classification Report (Validation Set):
              precision    recall  f1-score   support
    Healthy       0.95      0.97      0.96       200
Early Blight    0.94      0.93      0.93       180
Late Blight     0.96      0.94      0.95       190
```

**Accuracy: ~95%** across 3 classes (Healthy, Early Blight, Late Blight)

## 🏗️ Architecture
```
ResNet50V2 (pre-trained, partially frozen)
    ↓ GlobalAveragePooling2D
    ↓ Dense(512, ReLU) + Dropout(0.4)
    ↓ Dense(256, ReLU) + Dropout(0.3)  
    ↓ Dense(3, softmax)
```

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/yourusername/Potato-leaf-disease-classification-model-using-resnet50v2.git
cd Potato-leaf-disease-classification-model-using-resnet50v2

pip install -r requirements.txt
```

### 2. Download Dataset
Place your `PLD_3_Classes_256` folder with `Training/` and `Validation/` subdirectories in the root.

### 3. Train Model
```bash
python train.py  # Automatically trains + saves model
```

### 4. Launch Streamlit Demo
```bash
streamlit run app.py
```

## 🔧 Model Training Pipeline

```python
# Key optimizations implemented:
- GPU memory growth enabled
- Class weights for imbalanced data
- Two-phase training (frozen → fine-tune)
- Aggressive augmentation pipeline
- Learning rate reduction + early stopping
```

**Training took ~45 mins on RTX 3060** with 30+20 epochs.


## ⚙️ Requirements
```txt
tensorflow>=2.10
streamlit>=1.28
scikit-learn
numpy
pillow
```

## 🤝 Contributing
1. Fork the repo
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push & PR!

## 📄 Citation
```
Built for Unoteam Software Pvt Ltd internship project
Mansur Shaik - AI Engineer
```

## 🚀 Deployed Demo
🔗 [Live Streamlit App](https://your-streamlit-app-url.streamlit.app)

***

**Built with ❤️ for precision agriculture using Deep Learning**

<p align="center">
  <img src="assets/demo_screenshot.png" width="800"/>
</p>

***
