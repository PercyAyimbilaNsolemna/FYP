# 🧠 Decoder-Free DeepLabV3+ Inspired Swin-ViT for Kidney CT Image Classification  

## 📖 Overview  
This project investigates a **decoder-free DeepLabV3+ inspired Swin Vision Transformer (Swin-ViT)** framework for classifying kidney CT images into four categories: **Normal, Cyst, Tumor, and Stone**.  

Unlike traditional segmentation architectures that rely on heavy decoder modules, this approach removes the decoder stage and instead performs direct feature fusion between pretrained CNN backbones and the Swin-ViT encoder.  

The study demonstrates that eliminating the decoder reduces computational cost while maintaining high accuracy, offering a promising direction for **scalable and efficient diagnostic tools** in medical imaging.  

![Model Pipeline](path-to-your-pipeline-diagram.png)  
*Figure 1: Modified model pipeline showing encoder-only architecture.*  

---

## ⚙️ Installation and Setup  

### 🔹 Kaggle Notebook  
The project was implemented and tested on **Kaggle**, which provided both GPU and storage resources. To reproduce results on Kaggle:  
1. Open the notebook in Kaggle.  
2. Ensure the dataset path is set correctly.  
3. Click **Run All** to execute all cells.  

### 🔹 Local Environment (Optional)  
If running locally, install the required packages as listed in the first cell of the notebook:  

```bash
pip install torch torchvision torchaudio
pip install timm
pip install matplotlib numpy pandas scikit-learn
```

## 📂 Dataset  

The dataset used is the **CT KIDNEY DATASET: Normal-Cyst-Tumor and Stone**, publicly available on [Kaggle](https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone).  

- **Preprocessing**: Images were resized to **224×224**, normalized, and converted to tensors.  
- **Classes**: `Normal`, `Cyst`, `Tumor`, `Stone`.  
- All preprocessing and transformations are automatically handled in the notebook.  

## 🏗️ Model Architecture

The framework integrates:

- Pretrained CNN Backbones: VGG19, ResNet152V3, MobileNetV3.

- Patch Partition & Embedding using Swin Transformer.

- Hierarchical Representation with patch merging at multiple scales (2×2, 4×4, 6×6).

- Cross-contextual attention and adaptive pooling.

- Direct Feature Fusion of low- and high-level features, bypassing the decoder.

- Classification Head producing four output categories.

## 🚀 Training Configuration

- Batch Size: 10

- Epochs: 10

- Optimizer: Adam

- Loss Function: CrossEntropyLoss

- Transfer Learning: CNN backbones initialized with pretrained ImageNet weights.

- Training was conducted both with and without a decoder for comparison.

## 📊 Results  

The decoder-free model consistently outperformed the decoder-based version.  

### 🔹 Accuracy Comparison  

| Model                      | Training Accuracy | Testing Accuracy|  
|----------------------------|-------------------|-----------------|  
| With Decoder               | 97.61% (72.02 mins)| 99.20%         |  
| Without Decoder (Proposed) | 98.56% (41.69 mins)     | 99.72%        |  

### 🔹 Observations  
- The **decoder-free model** achieved higher testing accuracy and smoother convergence.  
- The **confusion matrix** showed strong performance in detecting normal and cyst classes, with occasional misclassifications between cyst and tumor due to structural similarities.  
- Training and validation loss curves confirmed stable optimization dynamics in the proposed architecture.  

![Accuracy Loss Curves](path-to-accuracy-loss-curve.png)  
*Figure 2: Training accuracy and loss curves for decoder-free model.*  

![Confusion Matrix](path-to-confusion-matrix.png)  
*Figure 3: Confusion matrix for decoder-free classification results.*  


## 🖥️ Usage  

Since this project is implemented in notebooks:  

1. Verify the dataset path is correct in the notebook.  
2. Run all cells sequentially, or simply select **Run All**.  
3. Two notebooks are included:  
   - `model_with_decoder.ipynb`  
   - `model_without_decoder.ipynb`  


## 🤝 Contributions

This repository is for academic and research purposes. Contributions are welcome via pull requests.

## 🙏 Acknowledgments  

- To **Almighty God** for guidance and strength. 

- To our **supervisor, Prof. Justice Kwame Appati**, for his mentorship and invaluable guidance.  

- To our **mentor, Mr Patrick Wunake**, for his continuous advice and encouragement.  

- To the **Data Intelligence and Swarm Analytics Laboratory (DISAL)**, University of Ghana, for providing the intellectual environment for research and collaboration.  

- To **Kaggle**, for providing both the dataset and the computational resources (GPU, storage) that enabled the successful training and evaluation of this model.  
