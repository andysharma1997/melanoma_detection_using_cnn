# Melanoma Detection using Convolutional Neural Network (CNN)
> This project focuses on building a custom convolutional neural network (CNN) model to accurately detect melanoma from skin lesion images. Melanoma, a type of cancer, accounts for 75% of skin cancer deaths, and an early detection solution can significantly aid dermatologists in reducing manual effort for diagnosis.

## Table of Contents
* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Conclusions](#conclusions)
* [Acknowledgements](#acknowledgements)
* [Contact](#contact)

<!-- You can include any other section that is pertinent to your problem -->

## General Information
- **Problem Statement**: To build a CNN-based model that can accurately detect melanoma, a deadly form of skin cancer. Early detection can reduce manual efforts in diagnosis and potentially save lives.
- **Dataset**: The dataset consists of 2,357 images of malignant and benign oncological diseases, provided by the International Skin Imaging Collaboration (ISIC). The dataset includes images of the following diseases:
  - Actinic keratosis
  - Basal cell carcinoma
  - Dermatofibroma
  - Melanoma
  - Nevus
  - Pigmented benign keratosis
  - Seborrheic keratosis
  - Squamous cell carcinoma
  - Vascular lesion

  The dataset is sorted according to ISIC's classifications and contains a relatively balanced set of images, with melanoma and moles being slightly dominant.

- **Business Problem**: The ability to detect melanoma early can drastically reduce skin cancer deaths by aiding dermatologists in faster, more accurate diagnoses.

## Project Pipeline
1. **Data Reading/Data Understanding**: Define paths for training and testing images.
2. **Dataset Creation**: Create train and validation datasets from the training directory, ensuring images are resized to 180x180 with a batch size of 32.
3. **Dataset Visualization**: Visualize one instance of each of the nine classes in the dataset.
4. **Model Building & Training**: 
   - Create a CNN model to classify the nine classes.
   - Rescale images to normalize pixel values between (0,1).
   - Choose an appropriate optimizer and loss function.
   - Train the model for approximately 20 epochs.
   - Analyze the model for overfitting or underfitting.
5. **Data Augmentation Strategy**: Implement augmentation strategies to address underfitting or overfitting.
6. **Model Training on Augmented Data**: Retrain the model using augmented data and analyze improvements.
7. **Class Distribution & Handling Imbalances**: 
   - Examine the class distribution in the training dataset, identifying which classes have the least and most samples.
   - Use the Augmentor library to rectify class imbalances.
8. **Model Training on Rectified Data**: 
   - Retrain the model on the balanced dataset for approximately 30 epochs and analyze the results.

## Conclusions
1. **First Iteration Results**: The model overfitted, achieving near 90% accuracy on training but only 50% on validation, indicating poor generalization.
2. **Second Iteration Results**: 
   - **Training Accuracy and Loss**: The training accuracy plateaus early, and loss doesn’t significantly decrease, suggesting ineffective learning.
   - **Validation Accuracy**: Remains low, highlighting the model's inability to generalize.
   - **Validation Loss**: High or slightly decreasing, indicating poor pattern recognition.
3. **Final Iteration Results**: The model improved significantly, with validation accuracy rising from 20% to 80%. The training and validation curves tracked closely, showing effective learning and minimal overfitting.
4. **Class Imbalance Impact**: Addressing class imbalances with augmentation and training on the balanced dataset helped improve the model's ability to generalize across all classes.

## Technologies Used
- TensorFlow == 2.18.0
- Keras == 3.8.0
- Augmentor==0.2.12
- Python==3.10.12
- NumPy==1.26.4


## Acknowledgements
Give credit here.
- This project was inspired by the need to aid dermatologists in detecting melanoma.

## Contact
Created by [Shubham Sharma](https://www.linkedin.com/in/shubham-sharma-andy) - feel free to contact me!
