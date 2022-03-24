# Required libraries
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.preprocessing.image import save_img
import os 
import numpy as np

# Image settings
TRAIN_DIR = "./Sentinel2LULC_354/Sentinel2LULC_354/"
N_TRAIN_IMAGES = 10266
AUG_TRAIN_DIR = "./Sentinel2LULC_354_augmented/Sentinel2LULC_354_augmented/"
MAX_AUG_IMAGES = 5

# Data generator to create new transformed images
aug_generator = ImageDataGenerator(
    rotation_range=45, 
    width_shift_range=0.15,
    height_shift_range=0.15, 
    shear_range=0.2, 
    zoom_range=[0.2, 0.5], 
    horizontal_flip=True, 
    fill_mode="nearest"
)

# Iterate over the train images to create new ones
for dir in os.listdir(TRAIN_DIR):
    for img in os.listdir(TRAIN_DIR+dir):
        new_img = load_img(TRAIN_DIR+dir+"/"+img)
        new_img = img_to_array(new_img)
        new_img = np.expand_dims(new_img, axis=0)
        aug_it = aug_generator.flow(
            new_img, 
            batch_size=1, 
            save_to_dir = AUG_TRAIN_DIR+dir,  
            save_prefix = img[:-4]+"_aug", 
            save_format ='jpg'
        )
        # Create 5 new images from each train image
        n_aug_images = 0
        for aug_image in aug_it:
            n_aug_images += 1
            if (n_aug_images == MAX_AUG_IMAGES):
                break