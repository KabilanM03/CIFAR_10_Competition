# **CIFAR-10 Competition: Advanced Neural Network Architecture**

This repository presents an advanced neural network architecture designed for **image classification** on the **CIFAR-10 dataset**. The project employs **deep learning techniques**, including **Residual Networks (ResNet)**, data augmentation, and **optimisation strategies** to enhance performance.

---

## **1. Project Overview**
This project is part of a **national-level CIFAR-10 competition**, aimed at developing a robust model for classifying images into **10 categories**. The architecture integrates **ResNet blocks**, **stochastic gradient descent (SGD)**, and **dynamic learning rate scheduling** to improve accuracy and generalisation.

---

## **2. Technology Stack**
The model is built using the following tools and frameworks:

- **Programming Language**: Python  
- **Deep Learning Framework**: PyTorch  
- **Data Processing**: NumPy, Pandas  
- **Visualisation**: Matplotlib, Seaborn  
- **Model Architecture**: ResNet (Residual Network)  

---

## **3. Data Preparation and Augmentation**
To enhance model generalisation, several **data augmentation techniques** are applied:

- **Random Cropping**: Crops images to **32×32 pixels** with **4-pixel padding**.
- **Random Horizontal Flip**: Flips images horizontally with a **50% probability**.
- **Colour Jitter**: Adjusts **brightness, contrast, saturation, and hue**.
- **Normalisation**: Scales pixel values to have a **mean of 0.5** and a **standard deviation of 0.5** per channel.

For the **testing dataset**, only **normalisation** is applied to ensure evaluation consistency.

---

## **4. Data Loading**
The **CIFAR-10 dataset** is imported using `torchvision.datasets.CIFAR10`. Data loaders handle batch processing efficiently:

- **Train Loader**: Shuffles training data and loads batches of **128 images**.
- **Test Loader**: Loads batches of **128 images** without shuffling.

---

## **5. Model Architecture**
The model follows a **ResNet-based approach**, consisting of:

### **ResNet Block**
To mitigate the **vanishing gradient problem**, the architecture employs **ResNet blocks**, each containing:

- **Convolutional Layers**: Two convolutional layers, each followed by **batch normalisation** and **ReLU activation**.
- **Skip Connections**: Enables residual learning, allowing gradients to flow smoothly through deeper layers.

### **Full ResNet Model**
The network consists of:

- **Initial Convolutional Layer**: Applies **batch normalisation** and **ReLU activation**.
- **Multiple ResNet Blocks**: With increasing **filter sizes**.
- **Fully Connected Layer**: Outputs **class probabilities**.

---

## **6. Training Pipeline**
The model is trained using **supervised learning** with the following configuration:

### **Loss Function**
- **Cross-Entropy Loss**: Used for **multi-class classification**.

### **Optimiser**
- **Stochastic Gradient Descent (SGD) with Momentum**:
  - Accelerates **gradient updates**, leading to **faster convergence**.

### **Learning Rate Scheduler**
- **ReduceLROnPlateau**:
  - Reduces the learning rate **when validation loss stagnates**, improving fine-tuning.

### **Gradient Clipping**
- **Prevents exploding gradients** by limiting gradient values during backpropagation.

### **Training Loop**
For each epoch, the following steps are executed:

1. **Forward Pass**: Processes inputs through the model.
2. **Loss Computation**: Evaluates the **cross-entropy loss**.
3. **Backward Pass**: Computes gradients using **backpropagation**.
4. **Optimisation Step**: Updates model parameters.
5. **Gradient Clipping**: Ensures **numerical stability**.

---

## **7. Model Evaluation**
At the end of each epoch, the model undergoes evaluation:

- **Accuracy Calculation**: Compares predicted labels with true labels.
- **Learning Rate Adjustment**: Adjusts learning rate **based on validation loss trends**.

---

## **8. Visualisation & Monitoring**
To monitor training progress, the following plots are generated:

- **Loss Plot**: Displays training loss over batches to assess model convergence.
- **Accuracy Plot**: Tracks training and testing accuracy over epochs.

---
