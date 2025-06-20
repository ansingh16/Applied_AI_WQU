# WorldQuant University – Applied AI Lab: Deep Learning for Computer Vision 🤖📸

Welcome to my repository documenting my journey through the **Applied AI Lab** offered by **WorldQuant University**. This lab explores the exciting intersection of deep learning and computer vision, where machines learn to interpret and understand images—just like humans do.

## About the Applied AI Lab

The Applied AI Lab is a fully online, tuition-free program designed by WorldQuant University to make advanced AI education accessible to learners around the world. The focus is on **practical, real-world applications**, giving participants the chance to build a strong foundation through hands-on projects.

Over the course of the lab, I’ll be working on six projects that dive deep into core computer vision topics, including:

* **Image Classification**: Teaching machines to recognize and categorize objects within images (e.g., distinguishing between cats and dogs). 🐶🐱
* **Object Detection**: Identifying and localizing multiple objects within a scene—essential for applications like autonomous vehicles. 🚗🚦
* **Generative AI**: Leveraging AI to create new images, from deepfakes to AI-generated art. 🎨

This repository will serve as a living record of my progress, reflections, and code throughout the lab. Stay tuned for updates and insights as I explore the cutting edge of AI in vision!




## Project 1: **Wildlife Conservation in Côte d'Ivoire**

In this project, I participated in a data science competition focused on supporting wildlife research. The objective was to classify animals captured in images from camera traps in a wildlife preserve, helping scientists monitor animal populations more efficiently.

To tackle this task, I developed and trained neural network models capable of analyzing images and predicting which animal species—if any—appears in each photo.

Key Learnings and Skills:
- Reading and preprocessing image data for machine learning tasks

- Using PyTorch to work with tensors and build deep learning models

- Designing and training a Convolutional Neural Network (CNN) optimized for image classification

- Making predictions on new, unseen images using the trained model

- Preparing model outputs for competition submission format

This project was a practical deep dive into computer vision techniques, combining real-world data with hands-on model development. It contains the following notebooks:

- Wildlife Conservation in Côte d'Ivoire/011-image-as-data.ipynb : contains code for PyTorch introduction, tensor and operations of tensors
-  Wildlife Conservation in Côte d'Ivoire/013-binary-classification.ipynb : Contain the code for binary classification model from the dataset provided.
- Wildlife Conservation in Côte d'Ivoire/014-multiclass-classification.ipynb: Multiclass model for classification of different wildlife species.

## Project2: Crop Disease in Uganda

Crop Disease Classification with CNNs and Transfer Learning
This project focuses on developing a computer vision model to classify crop disease images from a dataset collected in Uganda. A convolutional neural network (CNN) is trained to recognize five distinct categories of plant disease or health. To enhance model performance and generalization, the project incorporates transfer learning and a range of training optimization techniques.

Key Highlights:

- Performed exploratory data analysis on a labeled crop disease image dataset.

- Built and trained a convolutional neural network from scratch for image classification.

- Applied transfer learning by fine-tuning a pre-trained image classification model to adapt to the crop disease domain.

- Identified and addressed overfitting using regularization techniques.

- Evaluated model robustness using k-fold cross-validation.

- Optimized training with Keras callbacks, including learning rate scheduling, model checkpointing, and early stopping.

By the end of the project, the model achieves improved classification accuracy and demonstrates good generalization performance across folds.
The notebooks contains following notebooks:

- Crop Disease in Uganda/021-fix-my-code.ipynb: Common problems in Neural networks

- Crop Disease in Uganda/022-explore-dataset.ipynb: Exploration of Kaggle dataset

- Crop Disease in Uganda/023-multiclass-classification.ipynb:Multi-class classfication using the neural network. It highlights the problems of overfitting.

- Crop Disease in Uganda/024-transfer-learning.ipynb: Using Transfer learning to create a model to classify. It also contains the code for K-fold validation done for this model.


## Project 3: Traffic in Bangladesh

This project focuses on analyzing real-time traffic video feed data from Dhaka, Bangladesh. The main objective is to process video frames and detect objects such as cars, pedestrians, and other traffic-related entities using object detection techniques. You'll work with both pre-trained YOLO models and fine-tune them to detect custom objects relevant to the traffic scenes.

Key Highlights:

- Processed traffic video feed data from Dhaka, Bangladesh for object detection tasks.

- Extracted and labeled frames from raw video files for training and evaluation.

- Parsed XML annotation files containing bounding box data for custom object classes.

- Applied a pre-trained YOLO model for real-time object detection on traffic frames.

- Fine-tuned the YOLO model to detect custom traffic-related objects such as rickshaws and buses.

- Performed data augmentation to improve model generalization and reduce overfitting during training.

The notebooks contains following notebooks:

- Trafic in Bangladesh/032-traffic-data-as-images-and-video.ipynb: Dealing with images, videos, Parsing xml data using ElementTree.
- Trafic in Bangladesh/033-object-detection-with-yolov8.ipynb: Prediction using YOLO algorithm using pretrained model.



All the notebooks in the repository are licenced  by them to be used under the CC [BY-NC-ND 4.0 licence](https://creativecommons.org/licenses/by-nc-nd/4.0/).
