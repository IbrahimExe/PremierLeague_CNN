# Premier League Club Logo Recognition using CNN's!

As a passionate football fan and a supporter of Manchester United, I embarked on this project to deepen my understanding of Convolutional Neural Networks (CNNs) and their application in image recognition. This repository showcases a CNN model trained to identify English Premier League club logos, providing a practical example of how AI can be used in areas I'm enthusiastic about. This README serves as a comprehensive walkthrough of the project, from dataset selection and model architecture to training, evaluation, and potential real-world applications.

## What are Convolutional Neural Networks (CNNs)?

A Convolutional Neural Network (CNN) is a specialized type of neural network primarily used for analyzing visual imagery. Unlike traditional neural networks, CNNs are designed to automatically and adaptively learn spatial hierarchies of features from input images through a process called convolution. They are incredibly powerful for tasks such as:

*   **Image Classification:** Identifying the main object or category within an image (e.g., distinguishing between a cat and a dog).
*   **Object Detection:** Locating and classifying multiple objects within an image.
*   **Image Segmentation:** Dividing an image into segments to simplify its representation.
*   **Facial Recognition:** Identifying individuals from images or videos.

CNNs excel because they can learn to recognize patterns (like edges, textures, and shapes) at different levels of abstraction, making them highly effective for understanding complex visual data.

## Dataset: English Premier League Logos

### Dataset Details

For this project, I utilized the **English Premier League Logo Detection** dataset available on [Kaggle](https://www.kaggle.com/datasets/alexteboul/english-premier-league-logo-detection-20k-images/data). This dataset is ideal for a multi-class image classification task, featuring:

*   **20,000 images** in total.
*   **20 unique English Premier League club classes**, with 1,000 diverse images of each club's crest/logo.
*   Image dimensions are **140x140 pixels**, providing a good balance between detail and computational efficiency.

### Data Preprocessing and Augmentation

To prepare the images for training and to enhance the model's robustness, several preprocessing and data augmentation techniques were applied using `torchvision.transforms`:

*   **Random Rotation (10 degrees):** To expose the model to slight variations in logo orientation.
*   **Random Horizontal Flip:** To account for potential mirroring of logos.
*   **ColorJitter (brightness=0.2, contrast=0.2):** To simulate different lighting conditions and color variations.
*   **Resize (128x128):** Images were resized to a consistent 128x128 pixels, a common practice to standardize input for CNNs.
*   **ToTensor():** Converts PIL images or NumPy arrays to PyTorch tensors.
*   **Normalize:** Normalizes the images with standard mean and standard deviation values derived from ImageNet, which is beneficial for model training efficiency.

### Train/Test Split Rationale

A crucial step in machine learning is splitting the dataset into training and testing sets. For this project, an **80/20 split** was used (80% for training, 20% for testing).

*   **Training Set:** Used to train the CNN model, allowing it to learn patterns and features from the images. The model adjusts its internal parameters based on the errors it makes on this data.
*   **Test Set:** Used to evaluate the model's performance on unseen data. This provides an unbiased estimate of how well the model generalizes to new, real-world images it has never encountered during training. Without a separate test set, a model might simply memorize the training data (overfitting) and perform poorly on new examples.

`DataLoader` objects were used to efficiently load batches of images during training and testing, optimizing memory usage and speeding up the process.

## Model Architectures

To explore the impact of different network designs, I experimented with three variations of a Convolutional Neural Network architecture. Each variation modifies the number of layers, the number of filters, or the kernel sizes, influencing the model's capacity to learn features.

### Deep Network

This architecture aimed for increased depth by adding an additional convolutional block. It consists of three convolutional layers, each followed by a ReLU activation and max-pooling, leading to deeper feature extraction.

*   **Input Size:** (Batch_size, 3, 128, 128)
*   **Convolutional Blocks:**
    1.  `Conv2d(3, 6, kernel_size=3)` -> `MaxPool2d`
    2.  `Conv2d(6, 16, kernel_size=3)` -> `MaxPool2d`
    3.  `Conv2d(16, 32, kernel_size=3)` -> `MaxPool2d`
*   **Fully Connected Layers:** `Linear(32 * 14 * 14, 1000)` -> `Linear(1000, 500)` -> `Linear(500, 140)` -> `Linear(140, 20)`
*   **Trainable Parameters:** 6,852,148
