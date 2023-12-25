# Vision Transformer Models for Image and Time Series Classification

Welcome to the Vision Transformers (ViT) repository, dedicated to the exploration and implementation of Vision Transformer models for both Image Classification and Time Series Classification tasks.

## Overview
This repository focuses on leveraging Vision Transformer architectures, originally introduced for image classification, to achieve impressive results on diverse datasets. In particular, we have fine-tuned pre-trained ViT models using ImageNet weights on the Caltech101 dataset, achieving a remarkable 94% test accuracy across 102 classes.

## Getting Started
### Prerequisites
To get started with this repository, ensure you have the following dependencies installed:
- Python == 3.11
- PyTorch: torch == "2.1.0" & torchvision == "0.16.0"

### Usage
1. **Clone the Repository:**
    ```bash
    git clone https://github.com/Shyamal-Dharia/vision-transformers
    cd vision-transformers
    ```

2. **Train the ViT model for image classification**
    ```bash
    python train.py --epochs 5 --batch_size 32 --learning_rate 0.001 --data_folder "/path to your data folder/"
    ```
2. **Visualize the attention map**
    ```bash
    python visualize_attention_map.py --image_path "/path to your image/" --img_size 224
    ```

3. **Explore Models:**
    - Navigate to the `models` directory to access different Vision Transformer architectures (without pretrained weight).
    - Check out the architecture specifications and choose the one that best suits your task.

4. **Train model without pretrained weights:**
    - Check out `examples.py`, it has some examples on how to use ViT for images as well as time-series classification.


## Results
After fine-tuning ViT models on the Caltech101 dataset, we have achieved an impressive 94% test accuracy, showcasing the effectiveness of Vision Transformers in diverse classification tasks.

Training the model for 5 epochs.

![Alt text](https://github.com/Shyamal-Dharia/vision-transformers/blob/main/results.jpg?raw=true "Original image Vs Mean head")

Prediction from the model: cup

![Alt text](https://github.com/Shyamal-Dharia/vision-transformers/blob/main/overview.png?raw=true "Original image Vs Mean head")

![Alt text](https://github.com/Shyamal-Dharia/vision-transformers/blob/main/individual_heads.png?raw=true "Individual attention heads")

## References
 - https://www.learnpytorch.io/
 - https://keras.io/examples/vision/probing_vits/


## Contribution
Contributions to this repository are welcome! Whether you want to add new ViT architectures, share your fine-tuning results, or improve the documentation.

