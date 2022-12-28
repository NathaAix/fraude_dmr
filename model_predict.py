import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from tensorflow_hub.keras_layer import KerasLayer
import pathlib
import os
from PIL import Image
import pickle
import shutil
#import sklearn

#import yaml








IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_DEPTH = 3


    

    
def predict_image(Path_class,path, model):
    """Predict plane identification from image.
    
    Parameters
    ----------
    path (Path): path to image to identify
    model (keras.models): Keras model to be used for prediction
    
    Returns
    -------
    Predicted class
    """
    
    names= pd.read_csv(Path_class,names=['Names'])
    #A=np.array(Image.open(path))
    #r, g, b = image_rgb.getpixel((w, h))
    image= [np.array(Image.open(path).convert('RGB').resize((IMAGE_WIDTH, IMAGE_HEIGHT)))]
    #print(image)
    prediction_vector = model.predict(np.array(image))
    #print(prediction_vector)
    predicted_classes = np.argmax(prediction_vector, axis=1)[0]
    names_classes=names['Names'][predicted_classes]
    #print(names)
    #print(predicted_classes)
    return prediction_vector, predicted_classes,names_classes

def load_model(path):
    """Load tf/Keras model for prediction
    """
    return tf.keras.models.load_model(path)


file =[]
directory = r'/content/drive/MyDrive/FacturePdf/pdf'# change in function of our folder
file_destination = "/content/drive/MyDrive/factures fausses/facturePdf/image/"# change in function of our folder
Path_class = "classe_name.txt"
model = load_model("model_ame.h5")
model.summary()





for filename in os.listdir(directory):
    if filename.endswith(".pdf"):
        file.append(os.path.join(directory, filename))
        print(os.path.join(directory, filename))

    else:
        continue


        
For i in file:
    prediction_vector,prediction,classes = predict_image(Path_class,str(i), model)
    if classes == 'Facture_fausse':
        shutil.move(str(i), file_destination)
    else: print('Change the model')
    
    



    
