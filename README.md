# Numbers recognition
![sample_model_inference.png](sample_model_inference.png)

## Overview

This project aims to develop a machine learning model capable of recognizing and classifying handwritten digits. Unlike many basic models that focus on single-digit recognition, our project extends this functionality to handle multiple digits, making it suitable for more complex applications such as reading handwritten numbers from documents, forms, or images in the real world.

## Features

- **Data Preprocessing**: Includes scripts for cleaning and normalizing the input images.
- **Model Training**: Utilizes Convolutional Neural Networks (CNNs) to recognize patterns in digits.
- **Accuracy Improvement**: Techniques like data augmentation, dropout, and batch normalization to improve model accuracy.
- **Multi-Digit Recognition**: Specialized algorithms to handle sequences of digits.
- **User Interface**: A simple UI to upload images and view predictions.

## Prerequisites

- Python 3.8 or above
- TensorFlow 2.x
- NumPy
- Matplotlib (for visualization)
- OpenCV (for image processing)

## Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/Marcolino97do/Numbers_Recognition.git
```
Install the required packages:
```bash
pip install -r requirements.txt
```
Enter the source folder:
```bash
cd Numbers_Recognition/source
```
Execute the main.py file:
```bash
python3 main.py
```
After the first run, feel free to play with the parameters in the config module, 
and experiment with new modalities.

## Usage

## Dataset Creation
