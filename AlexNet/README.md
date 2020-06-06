# Sample code of AlexNet implemented on Tensorflow 1, along with my review on AlexNet

## Instructions for use
- Download the files: alexNet.ipynb, alexNet.py and cifar10_utils.py into the same directory in your system.
- Run the file alexNet.ipynb on your Jupyter Notebook or on Google Colab or in any other similar platform.
- The appropriate directories will be auto-created when the code runs.
- Has provisions to start training from the last checkpoint.
- Testing can be done on the entire test dataset, or on randomly selected k images, or simply a quick prediction on a single image. A sample prediction is shown below:
![cifar10_pred](https://github.com/sandipan211/DL-templates/blob/master/AlexNet/test%20pred.png)

## Requirements:
- Tensorflow
- Numpy
- Skimage (for image resizing)
- Tqdm
- Pickle

## Runtime: 
About 1 hour 33 minutes on Google Colab

## About the .py files:
### 1. alexNet.py
- Contains the primary model architecture along with training and testing facilities
- **Model utilities**:
   - Dataset used: CIFAR10 (although the code can be made extensible to work for other datasets as well)
   - Epochs: 15 (maybe less if early stopping occurs)
   - Batch size: 64
   - Learning rate: 0.00005
   - Validation data percentage: 10% of the training dataset
   - Input image dimensions: 227x227x3
- For more details on the implemented architecture, go through the architecture() method in alexNet.py, and you can also read my study on AlexNet.

### 2. cifar10_utils.py
- Downloads the data (if it does not exist already), displaying a tqdm progress bar. If extracted dataset is not present in the directory, then extracts it into the 'cifar-10-batches-py' folder.

- Preprocesses the training, validation and testing sets and stores them into pickle files.

- Has utility functions to resize data into batches of 227x227x3 and returns them to calling functions.

