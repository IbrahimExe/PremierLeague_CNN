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

### Wide Network

The Wide Network focused on increasing the number of filters (channels) in each convolutional layer, making it wider.
This allows the model to learn a broader range of features at each convolutional step.

*   **Input Size:** (Batch_size, 3, 128, 128)
*   **Convolutional Blocks:**
    1.  `Conv2d(3, 12, kernel_size=3)` -> `MaxPool2d`
    2.  `Conv2d(12, 32, kernel_size=3)` -> `MaxPool2d`
    3.  `Conv2d(32, 64, kernel_size=3)` -> `MaxPool2d`
*   **Fully Connected Layers:** `Linear(64 * 14 * 14, 1000)` -> `Linear(1000, 200)` -> `Linear(200, 20)`
*   **Trainable Parameters:** 12,771,540

### Kernel Network

This variation experimented with different kernel sizes, specifically using 5x5 kernels instead of the standard 3x3. Larger kernels can capture more global features in early layers, potentially affecting the receptive field of the convolutions.

*   **Input Size:** (Batch_size, 3, 128, 128)
*   **Convolutional Blocks:**
    1.  `Conv2d(3, 6, kernel_size=5)` -> `MaxPool2d`
    2.  `Conv2d(6, 16, kernel_size=5)` -> `MaxPool2d`
    3.  `Conv2d(16, 32, kernel_size=5)` -> `MaxPool2d`
*   **Fully Connected Layers:** `Linear(32 * 12 * 12, 1000)` -> `Linear(1000, 200)` -> `Linear(200, 20)`
*   **Trainable Parameters:** 4,828,924

## Training and Evaluation

Each model was trained for 10 epochs using `nn.CrossEntropyLoss` as the loss function. Different optimizers and learning rates were chosen for each variation:

*   **Deep Network:** `Adam` optimizer with a learning rate of `0.001`.
*   **Wide Network:** `SGD` optimizer with a learning rate of `0.001` and `momentum=0.9`.
*   **Kernel Network:** `Adam` optimizer with a learning rate of `0.01`.

The training process involved iterating through the `train_loader`, computing the loss, performing backpropagation, and updating model weights. The `evaluate` function assessed the model's performance on the `test_loader` without updating weights, providing validation metrics.

## Results and Analysis

### Training Curves

The training and validation loss and accuracy curves provided critical insights into each model's learning process:

*   **Deep Network:** Showed excellent convergence, with both training and validation loss decreasing rapidly and accuracy quickly approaching 100%. This indicates effective learning and good generalization.
<div align="center">
    <img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/987ed463-4e45-463d-a4ad-245192665790" />
</div>


*   **Wide Network:** Exhibited very slow learning, with both loss and accuracy improving only marginally over 10 epochs. This suggests the chosen architecture, hyperparameters (especially the optimizer or learning rate), or both, were not well-suited for this dataset.
<div align="center">
    <img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/976e6448-69a9-49c4-8fb1-00f396169964" />
</div>


*   **Kernel Network (5x5):** Similar to the Wide Network, this model struggled significantly. The loss remained high, and accuracy showed minimal improvement, indicating a failure to learn meaningful features.
<div align="center">
    <img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/fb604d36-b504-47cc-9e65-83486056458c" />
</div>


**Conclusion from Curves:** The **Deep Network** was by far the best performer among the three, quickly achieving high accuracy and low loss, indicating robust learning and generalization capabilities for the Premier League logo classification task.

### Confusion Matrix

Focusing on the best-performing **Deep Network**, a confusion matrix was generated to visualize the model's classification performance across all 20 classes:

<div align="center">
    <img width="1400" height="1450" alt="image" src="https://github.com/user-attachments/assets/69470dde-b374-43c9-98ea-bb2ee05022b1" />
</div>

The confusion matrix clearly demonstrates the exceptional accuracy of the Deep Network. The diagonal elements, representing correct classifications, show very high numbers, while off-diagonal elements are extremely sparse, indicating minimal misclassifications. This matrix confirms that the model is highly effective at distinguishing between the 20 different Premier League club logos.

### Misclassified Examples

Despite the Deep Network's high overall accuracy, a few images were still misclassified. Analyzing these specific examples can offer valuable insights into the model's limitations or challenges within the dataset. For instance, some logos might share similar color schemes, shapes, or be visually ambiguous due to image quality or angle. In this case, only 6 images out of 4,000 test images were misclassified:

<div align="center">
    <img width="1607" height="1109" alt="image" src="https://github.com/user-attachments/assets/3fe98929-fb6d-485f-9475-f0eec3c9fd08" />
</div>

These rare misclassifications highlight that even highly accurate models can occasionally struggle with subtle visual similarities or imperfections in the input data.

## Conclusion and Potential Applications

This project successfully demonstrated the power of Convolutional Neural Networks for image recognition, specifically applied to classifying English Premier League club logos. The **Deep Network** architecture proved to be the most effective, achieving near-perfect accuracy on the test set. Through experimentation, I learned the critical importance of architectural design and hyperparameter tuning for successful model training.

Learning about and building such a system has numerous potential applications and broader contexts in AI:

*   **Automated Sports Analysis:** Automatically identifying team logos in broadcast footage, fan-submitted content, or merchandise.
*   **Brand Recognition:** Extending this concept to general brand recognition for marketing analytics or counterfeit detection.
*   **Visual Search:** Enabling users to search for products or information by simply uploading an image.
*   **Content Moderation:** Automatically identifying and categorizing visual content on platforms.
*   **Educational Tools:** Creating interactive tools for sports enthusiasts or collectors to identify club logos.

This project not only solidified my understanding of CNNs but also showcased how a personal passion can drive engaging and educational AI development.

## Resources

*   **Kaggle Dataset:** [English Premier League Logo Detection](https://www.kaggle.com/datasets/alexteboul/english-premier-league-logo-detection-20k-images/data)
*   **Google Colab Notebook:** [Colab Notebook](https://drive.google.com/file/d/1xif3u9zh_wJ7uOpTZ-fKgZvtEUYXMLLZ/view?usp=sharing)
*   **GitHub Repository:** [Current Repo](https://github.com/IbrahimExe/PremierLeague_CNN)
