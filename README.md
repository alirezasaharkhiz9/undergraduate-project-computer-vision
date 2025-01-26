# Undergraduate Project - Computer Vision

### Overview

This undergraduate project focuses on exploring and developing advanced techniques in computer vision, structured into two main sections: Classical Computer Vision and Modern Deep Learning-based Computer Vision. The project begins by covering the fundamentals of image processing, video analysis, and classical object detection algorithms. In the second part, using cutting-edge tools like PyTorch, convolutional neural networks (CNN) will be implemented. The project also delves into modern techniques such as transfer learning, pre-trained models, and the emerging Vision Transformers. Furthermore, efficient object detection with YOLO and facial recognition using deep learning will be thoroughly explored.

### Table of Contents

1.  [Project Structure](#project-structure)
2.  [Modern CV vs Classical CV](#modern-cv-vs-classical-cv)
3.  [Roadmap](#roadmap)
4.  [Requirements](#requirements)
5.  [References](#references)

### Project Structure

The project will be organized into the following sections:
#### 1. [Classical Computer Vision](https://github.com/alirezasaharkhiz9/undergraduate-project-computer-vision/tree/main/Classical%20Computer%20Vision)
- Image Processing
  - [Image processing](https://github.com/alirezasaharkhiz9/undergraduate-project-computer-vision/blob/main/Classical%20Computer%20Vision/ImageProcessing.ipynb)
- Video Processing
  - [Working with Video](https://github.com/alirezasaharkhiz9/undergraduate-project-computer-vision/blob/main/Classical%20Computer%20Vision/WorkingWithVideo.ipynb)
- Object Detection
  - [Object Detection](https://github.com/alirezasaharkhiz9/undergraduate-project-computer-vision/blob/main/Classical%20Computer%20Vision/ObjectDetection.ipynb)
- Face Detection
  - [Face Detection](https://github.com/alirezasaharkhiz9/undergraduate-project-computer-vision/blob/main/Classical%20Computer%20Vision/FaceDetection.ipynb)
- OCR
  - [Text Detection](https://github.com/alirezasaharkhiz9/undergraduate-project-computer-vision/blob/main/Classical%20Computer%20Vision/TextDetection.ipynb)
#### 2. [Modern Computer Vision - Deep Learning](https://github.com/alirezasaharkhiz9/undergraduate-project-computer-vision/tree/main/Modern%20Computer%20Vision)
- Image Classification
  - [Cnn Model theory](https://github.com/alirezasaharkhiz9/undergraduate-project-computer-vision/blob/main/Modern%20Computer%20Vision/CnnModelTheory.ipynb)
  - [Cnn With Keras (Normal, Pneumonia and Tuberculosis Lungs)](https://github.com/alirezasaharkhiz9/undergraduate-project-computer-vision/blob/main/Modern%20Computer%20Vision/CnnWithKeras.ipynb)
  - [CNN With Pytorch Lightning (Fundus Glaucoma Detection)](https://github.com/alirezasaharkhiz9/undergraduate-project-computer-vision/blob/main/Modern%20Computer%20Vision/CnnWithPytorchLightning.ipynb)
  - [Residual Network With Pytorch Lightning (Normal, Pneumonia and Tuberculosis Lungs)](https://github.com/alirezasaharkhiz9/undergraduate-project-computer-vision/blob/main/Modern%20Computer%20Vision/ResidualNetworkWithPytorchLightning.ipynb)
- Image Manipulation
  - [AutoEncoder With Keras (The main objective of creating this dataset is to create autoencoder network that can colorized grayscale landscape images)](https://github.com/alirezasaharkhiz9/undergraduate-project-computer-vision/blob/main/Modern%20Computer%20Vision/AutoEncoderWithKeras.ipynb)
- Image Generation
  - [DCGAN With Keras (GAN using the MNIST dataset)](https://github.com/alirezasaharkhiz9/undergraduate-project-computer-vision/blob/main/Modern%20Computer%20Vision/DCGANWithKeras.ipynb)
- Image Segmentation
  - [Lungs Segmentation Using U-Net architecture](https://github.com/alirezasaharkhiz9/undergraduate-project-computer-vision/blob/main/Modern%20Computer%20Vision/LungsSegmentationUsingU_Net.ipynb)
  - [Tumor Segmentation Using Yolo v11](https://github.com/alirezasaharkhiz9/undergraduate-project-computer-vision/blob/main/Modern%20Computer%20Vision/TumorSegmentationUsingYolo.ipynb)
- Object Detection
  - [Object Detection Using YOLO v5 (road-detection)](https://github.com/alirezasaharkhiz9/undergraduate-project-computer-vision/blob/main/Modern%20Computer%20Vision/ObjectDetectionUsingYOLOv5.ipynb)
  - [Object Detection Using YOLO v8 (Bone Fracture Detection)](https://github.com/alirezasaharkhiz9/undergraduate-project-computer-vision/blob/main/Modern%20Computer%20Vision/ObjectDetectionUsingYOLOv8.ipynb)
  - [Object Detection Using YOLO v11 (Tumor Detection)](https://github.com/alirezasaharkhiz9/undergraduate-project-computer-vision/blob/main/Modern%20Computer%20Vision/TumorDetectionUsingYolov11.ipynb)
- Object tracking
  - [Object Tracking With YOLO v11](https://github.com/alirezasaharkhiz9/undergraduate-project-computer-vision/blob/main/Modern%20Computer%20Vision/ObjectTrackingWithYolo.ipynb) - [Download Result Video](https://github.com/alirezasaharkhiz9/undergraduate-project-computer-vision/blob/main/Modern%20Computer%20Vision/ObjectTrackingWithYolo.avi)
- pose estimation
  - ?
- Face Recognition
  - ?

### Modern CV vs Classical CV

| **Topic**                          | **Classical Computer Vision**                                                | **Modern Computer Vision (Using Deep Learning)**                                                                |
|-----------------|-----------------------|---------------------------------|
| **Definition and Core Principles** | Based on hand-crafted algorithms and engineered features.                    | Based on deep neural networks and learning from large datasets.                                                 |
| **Feature Extraction Method**      | Features are extracted manually using algorithms like SIFT, SURF, and HOG.   | Features are automatically learned by convolutional neural networks (CNNs).                                     |
| **Accuracy**                       | Accuracy is usually limited and optimization for complex problems is harder. | Generally achieves much higher accuracy, especially in complex tasks like image recognition and classification. |
| **Amount of Data Required**        | Requires less data, but models are usually less optimized.                   | Requires large amounts of data for effective learning.                                                          |
| **Computational Complexity**       | Generally lighter and simpler in computation.                                | Heavier in computation and requires powerful hardware (GPUs).                                                   |


### Roadmap

The project will progress through the following phases:
1. **Phase 1**: Implementing Classical Computer Vision Techniques
2. **Phase 2**: Building and Training Deep Learning Models with PyTorch and tensorflow, keras


### Requirements

To run the project, you'll need:

-   Python 3.x
-   tools: Jupyter Lab, Colab, lightning studio
-   Required libraries listed in requirements.txt.

You can install the dependencies using:

``` bash
pip install -r requirements.txt
```

### References:

- Ayyadevara, V. K., & Reddy, Y. (2024). Modern Computer Vision with PyTorch - Second Edition: A practical roadmap from deep learning fundamentals to advanced applications and Generative AI (2nd ed.). Packt Publishing.
- Elgendy, M. (2020). Deep Learning for Vision Systems (1st ed.). Manning.
- Ratan, R. D. (2024). Modern Computer Vision GPT, PyTorch, Keras, OpenCV4 in 2024! Next-Gen Computer Vision: YOLOv8, DINO-GPT4V, OpenCV4, Face Recognition, GenerativeAI, Diffusion Models & Transformers [Online course]. Udemy.
