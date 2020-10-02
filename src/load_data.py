import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from keras.preprocessing.image import ImageDataGenerator
import os

def load_data(dataset_id):
    X = pd.read_csv(f'../data/X{dataset_id}.csv').values
    y = pd.read_csv(f'../data/y{dataset_id}.csv').values
    return X, y

def load_text(dataset_id):
    X = pd.read_csv(f'../data/X{dataset_id}.csv').fillna('').iloc[:, 0].values
    y = pd.read_csv(f'../data/y{dataset_id}.csv').values
    return X, y

def load_images(dataset_id, image_size):
    datagen = ImageDataGenerator(
        rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True, 
    )
    
    color_mode = 'grayscale' if image_size[2] == 1 else 'rgb'
    class_mode = 'binary' if len(os.listdir(f'../data/{dataset_id}/training/')) == 2 else 'categorical'
    
    training_set = datagen.flow_from_directory(
        f'../data/{dataset_id}/training/', target_size=image_size[:2],  batch_size=32,
        class_mode=class_mode, color_mode=color_mode
    )
    validation_set = datagen.flow_from_directory(
        f'../data/{dataset_id}/validation/', target_size=image_size[:2],  batch_size=32,
        class_mode=class_mode, color_mode=color_mode
    )
    print(training_set.class_indices)
    return training_set, validation_set

def create_image_sets(dataset_id, image_size):
    datagen = ImageDataGenerator(
        rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True, 
        validation_split=0.1,
    )
    
    color_mode = 'grayscale' if image_size[2] == 1 else 'rgb'
    class_mode = 'binary' if len(os.listdir(f'../data/{dataset_id}/training/')) == 2 else 'categorical'
    
    training_set = datagen.flow_from_directory(
        f'../data/{dataset_id}/', target_size=image_size[:2],  batch_size=32, class_mode=class_mode, 
        subset='training', color_mode=color_mode
    )
    validation_set = datagen.flow_from_directory(
        f'../data/{dataset_id}/', target_size=image_size[:2],  batch_size=32, class_mode=class_mode, 
        subset='validation', color_mode=color_mode
    )
    print(training_set.class_indices)
    return training_set, validation_set