import os, random
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

class ImageAugmenter():
    def __init__(self, imagegen_obj, n_images=None, cmap='viridis'):
        self.imagegen_obj = imagegen_obj
        self.n_images = n_images
        self.cmap = cmap

    def fit(self, source_directory):
        self.source_directory = source_directory
        
    def transform(self, target_directory):
        self.target_directory = target_directory
        training_directory = os.path.join(target_directory, 'training')
        validation_directory = os.path.join(target_directory, 'validation')
        class_examples = []
        
        if not os.path.exists(self.target_directory):
            os.mkdir(self.target_directory)
            os.mkdir(training_directory)
            os.mkdir(validation_directory)
            for directory in os.listdir(self.source_directory):
                os.mkdir(os.path.join(training_directory, directory))
                os.mkdir(os.path.join(validation_directory, directory))
                
        for directory in os.listdir(self.source_directory):
            class_examples.append(len(os.listdir(os.path.join(self.source_directory, directory))))
            
        print('Class Distribution:', class_examples)
        
        if self.n_images == None:
            self.n_images = max(class_examples)
            
        for i, directory in enumerate(os.listdir(self.source_directory)):
            training_dest_dir = os.path.join(training_directory, directory)
            validation_dest_dir = os.path.join(validation_directory, directory)
            if self.n_images > class_examples[i]:        
                q, r = divmod(self.n_images, class_examples[i])
                q, r = q-1, r/class_examples[i]
                for imgpath in tqdm(os.listdir(os.path.join(self.source_directory, directory))):
                    filename, file_extension = os.path.splitext(imgpath)
                    img = plt.imread(os.path.join(self.source_directory, directory, imgpath))
                    validation_prob = random.random()
                    if validation_prob <= (self.n_images/class_examples[i])*0.1:
                        plt.imsave(os.path.join(validation_dest_dir, imgpath), img, cmap=self.cmap)
                    else:
                        plt.imsave(os.path.join(training_dest_dir, imgpath), img, cmap=self.cmap)
                    for i in range(q):
                        if self.cmap == 'gray':
                            aug_img = np.squeeze(self.imagegen_obj.random_transform(np.expand_dims(img, axis=2)))
                        else:
                            aug_img = imagegen_obj.random_transform(img)
                        plt.imsave(os.path.join(training_dest_dir, f"{filename}-{i+1}{file_extension}"), aug_img, cmap=self.cmap)
                    prob = random.random()
                    if prob <= r:
                        if self.cmap == 'gray':
                            aug_img = np.squeeze(self.imagegen_obj.random_transform(np.expand_dims(img, axis=2)))
                        else:
                            aug_img = imagegen_obj.random_transform(img)
                        plt.imsave(os.path.join(training_dest_dir, f"{filename}-{0}{file_extension}"), aug_img, cmap=self.cmap)
            else:
                r = self.n_images/class_examples[i]
                for imgpath in tqdm(os.listdir(os.path.join(self.source_directory, directory))):
                    filename, file_extension = os.path.splitext(imgpath)
                    img = plt.imread(os.path.join(self.source_directory, directory, imgpath))
                    prob = random.random()
                    if prob <= r:
                        validation_prob = random.random()
                        if validation_prob <= 0.1:
                            plt.imsave(os.path.join(validation_dest_dir, imgpath), img, cmap=self.cmap)
                        else:
                            plt.imsave(os.path.join(training_dest_dir, imgpath), img, cmap=self.cmap)
            
            training_class_examples = [
                len(os.listdir(os.path.join(training_directory, directory))) for directory in os.listdir(training_directory)
                ]
            validation_class_examples = [
                len(os.listdir(os.path.join(validation_directory, directory))) for directory in os.listdir(validation_directory)
                ]  
            print('Updated Class Distribution\nTraining:', training_class_examples, '\nValidation:', validation_class_examples)
            
    def fit_transform(self, source_directory, target_directory):
        self.fit(source_directory)
        self.transform(target_directory)