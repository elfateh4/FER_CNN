# Facial Emotion Recognition: Two Approaches

## Overview

This project focuses on building a model capable of predicting and classifying human emotions based on facial expressions. Over the past century, researchers have worked to develop tools and frameworks to understand human emotions through facial expressions. Notable approaches include the Circumplex Model of Affect, the Facial Action Coding System, and large-scale datasets. This project leverages large datasets to develop an accurate classification model for facial emotion recognition.

## Team Members
- Mohammed El-Fateh Sabry
- Mostafa Ahmed
- Ahmed Abdelwahab
- Ahmed Ashour
- Hatem Mohamed
- Mohammad Hesham
- Mohamed Adel

## Project Components

- **Flask Web App**: Provides a user interface for interacting with the models.
- **CNN Model** (ResEmoteNet): A deep convolutional neural network for emotion classification.
- **Vision Transformer (ViT)**: A transformer-based model for image processing and classification.
- **Face Detection (RetinaFace)**: Detects faces in input images.
- **Decision Calculator**: Combines the results from the classification models for improved accuracy.

## Dataset Description

We used the **FER2013** dataset for training and testing our model. The dataset contains 48x48 pixel grayscale images of faces, automatically aligned and categorized into seven emotion classes:
- **Anger**
- **Disgust**
- **Fear**
- **Happy**
- **Sad**
- **Surprise**
- **Neutral**

### Dataset Details:
- **Training Set**: 28,709 examples
- **Test Set**: 3,589 examples

The faces are categorized based on the emotion shown in the facial expression. The dataset is divided into seven emotion categories and contains training and testing data, with various file counts for each emotion.

## Approach

### Face Detection

To optimize computational efficiency, we used **RetinaFace**, a face detection model that allows us to extract faces from large images and pass them to the emotion classification models. Despite its relatively longer run time compared to other models, RetinaFace was chosen for its higher accuracy and ease of implementation.

### Classification Models

#### 1. **ResEmoteNet**

After evaluating various models on the **Papers with Code** leaderboard, we selected **ResEmoteNet** due to its superior accuracy. We decided to retrain the model instead of using the original checkpoints due to differences in label encoding and to assess the training process ourselves. Retraining allowed us to tweak the model for better accuracy.

#### 2. **Vision Transformer (ViT)**

Although many models used CNNs, we explored the **Vision Transformer (ViT)** as a potential approach for our project. The Vision Transformer (ViT) was pretrained on **ImageNet-21k** and fine-tuned on **ImageNet 2012**. Since the ViT requires images of size 224x224, we adapted our dataset's 48x48 images by resizing them to the required dimensions.

### Decision Calculator

To combine the results of the two classification models, we built a simple **Decision Calculator**. The process involves:
1. Multiplying the accuracy of each model by its corresponding classification result.
2. Adding the weighted results of both models.
3. Calculating the average of the weighted results.
4. Based on the average result, determining the best class for the image.

The final decision is based on the class with the highest average result:
\[
\text{Best Class} = \arg\max (\text{Average Result})
\]

## User Interface

The user interface of the project provides several options for input and output:

### Input Options:
- **Upload Images** (Completed)
- **Take Images** (Completed)
- **Live Classification** (Coming soon)

### Output Options:
- **Suggestions for actions based on classification** (Coming soon)
- **Face images with their affiliation percentages for each emotion class** (Coming soon)

## Conclusion

This project leverages advanced models and techniques to accurately predict and classify emotions based on facial expressions. By combining the results from ResEmoteNet and Vision Transformer, we achieve improved accuracy in emotion recognition.

## Resources

- Vision Transformer: [Fine-tuning the Vision Transformer on CIFAR-10 with PyTorch Lightning](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/VisionTransformer/Fine_tuning_the_Vision_Transformer_on_CIFAR_10_with_PyTorch_Lightning.ipynb)
- Introduction to Vision Transformer (ViT): [Viso.ai](https://viso.ai/deep-learning/vision-transformer-vit/)
- Hugging Face Documentation for ViT: [Hugging Face Docs](https://huggingface.co/docs/transformers/en/model_doc/vit)
