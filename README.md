
### Machine Learning and Deep Learning Notebooks

This repository contains a collection of Jupyter notebooks exploring various machine learning and deep learning concepts, primarily using PyTorch and scikit-learn.

Notebooks
1. Computer_vision_CNN.ipynb
This notebook focuses on Convolutional Neural Networks (CNNs) for computer vision tasks. It likely covers:

Building and training CNN models.

Applying CNNs to image classification or other computer vision problems.

Demonstrations of CNN architectures and their implementation.

Performance:
Train loss: 0.32311 | Train accuracy: 88.25%
Test loss: 0.32619 | Test accuracy: 88.14%

2. Mnist.ipynb
This notebook is dedicated to working with the MNIST dataset, a classic benchmark for image classification. It includes:

Loading and preparing the MNIST dataset.

Implementing and training a model (likely a neural network) to classify handwritten digits.

Evaluating the model's performance on the MNIST dataset.

Performance:
Train Loss = 0.0006, Train Acc = 0.9994, Test Acc = 0.9753



3. Non linear dataset.ipynb
This notebook explores machine learning models designed for non-linear datasets. Key aspects include:

Generating non-linear datasets (e.g., using make_circles from scikit-learn).

Visualizing non-linear data distributions.

Developing and evaluating models capable of classifying or regressing on non-linear patterns.

The notebook uses sklearn and matplotlib to create and visualize data.

4. pytorch Model Practice.ipynb
This notebook serves as a general practice ground for fundamental PyTorch model concepts. It covers essential elements of building and training models in PyTorch, such as:

Defining neural network architectures using torch.nn.

Creating datasets and data loaders.

Implementing training loops.

Saving and loading trained models.

It demonstrates basic linear regression setup with known parameters, data creation, and train/test splitting.

Here are some key pointers for the Finetuning_LayoutLMv2_ForTokenClassification_FUNSD_fixed.ipynb code file:

5)Finetuning_LayoutLMv2_ForTokenClassification_FUNSD_fixed.ipynb

Key Libraries & Dependencies 📦
The notebook relies on several crucial libraries for deep learning, natural language processing, and computer vision tasks:

datasets: For easily loading and managing datasets, specifically the FUNSD dataset from Hugging Face.

transformers: Essential for using and finetuning the LayoutLMv2 model.

seqeval: Used for evaluating sequence labeling tasks, common in token classification.

torch, torchvision, torchaudio: PyTorch components for deep learning operations, especially for handling images and tensors.

detectron2: A library for object detection and segmentation, which might be used internally by LayoutLMv2 for its vision components.

PIL (Pillow): For image manipulation and processing (e.g., converting images to RGB, drawing bounding boxes).

Dataset Used 📊
FUNSD Dataset: A publicly available dataset specifically for form understanding in noisy scanned documents.

Structure: It contains words, bboxes (bounding box coordinates for each word), ner_tags (named entity recognition labels), and image data.

Splits: The dataset is typically divided into training (149 examples) and testing (50 examples) sets.

NER Labels: The ner_tags are mapped to labels such as O (Outside), B-HEADER, I-HEADER, B-QUESTION, I-QUESTION, B-ANSWER, and I-ANSWER.

Main Steps & Sections 🛠️
The notebook generally follows these key steps:

Installation of Dependencies: Installing all necessary Python packages.

Dataset Loading: Loading the FUNSD dataset from Hugging Face.

Data Preprocessing:

Defining id2label and label2id mappings for NER tags.

Preparing the data for the LayoutLMv2 model, which involves tokenization, creating attention masks, and incorporating bounding box information.

Model Initialization: Loading a pre-trained LayoutLMv2 model suitable for token classification.

Model Finetuning: Training the LayoutLMv2 model on the preprocessed FUNSD training data.

Evaluation: Assessing the finetuned model's performance on the test set using metrics relevant to token classification (accuracy, precision, recall, F1-score).

overall_precision': 0.7780441035474592|overall_recall: 0.814350225790266|overall_f1: 0.7957832802157392|overall_accuracy: 0.7952369834228344

### Key Functions Explained:

Dataloader (Data Handling):

In pytorch Model Practice.ipynb, data is prepared by creating a simple linear dataset and then splitting it into training and testing sets (X_train, y_train, X_test, y_test) using an 80/20 ratio. This process effectively simulates a dataset for the model to learn from and evaluate on. While a formal torch.utils.data.DataLoader is not explicitly used, the data is structured and prepared in a way that serves the same purpose of providing inputs (X) and corresponding labels (y) for model training and evaluation.

Trainer (Training Loop):

The training process in this notebook involves a custom linear regression model (LinearRegressionModel) which is an nn.Module subclass. The training loop, although not explicitly named "trainer," encompasses the steps where the model learns from the data. This involves defining an optimizer (e.g., Stochastic Gradient Descent) and a loss function (e.g., Mean Absolute Error). During training, the model iterates through the training data, calculates the loss between predictions and actual values, and then updates its parameters (weights and biases) using backpropagation and the optimizer to minimize this loss. The forward pass of the model is defined within the forward method of the LinearRegressionModel class, which computes self.weights * x + self.bias.

Installation and Setup
To run these notebooks, you'll need to have Python and Jupyter installed, along with the necessary libraries.

Clone the repository:

```bash
    git clone <repository_url>
    cd <repository_name>
```

B

```bash
    # Create a virtual environment
    python -m venv venv

    # Activate the virtual environment (depending on your OS)
    source venv/bin/activate   # Linux/macOS
    venv\Scripts\activate.bat  # Windows (for cmd)
    venv\Scripts\Activate.ps1  # Windows (for PowerShell)
```
Install the required libraries:
While a requirements.txt is not provided, based on the notebooks, you will likely need:

```bash

pip install torch torchvision scikit-learn matplotlib pandas
For Google Colab specific functionalities, no local installation is needed as it's a cloud-based environment.
```

