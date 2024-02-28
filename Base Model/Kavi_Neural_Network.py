import numpy as NP
import matplotlib.pyplot as PLT
import tensorflow as TF


#load data
MNIST_Dataset = TF.keras.datasets.mnist
(Image_Data_Training, Classification_Data_Training), (Image_Data_Evaluating, Classification_Data_Evaluating) = MNIST_Dataset.load_data()

#normalize data
Image_Data_Training = Image_Data_Training / 255.0
Image_Data_Evaluating = Image_Data_Evaluating  / 255.0

Image_Data_Training = Image_Data_Training.reshape(Image_Data_Training.shape[0], -1).T
Image_Data_Evaluating = Image_Data_Evaluating.reshape(Image_Data_Evaluating.shape[0], -1).T

