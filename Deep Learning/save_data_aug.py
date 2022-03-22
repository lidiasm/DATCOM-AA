from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.preprocessing.image import save_img
import os 

# Image settings
BATCH_SIZE = 128
HEIGHT = 224
WIDTH = 224
CHANNELS = 3
N_CLASSES = 29
TRAIN_DIR = "./Sentinel2LULC_354/Sentinel2LULC_354/"
AUG_TRAIN_DIR = "./Sentinel2LULC_354_augmented/Sentinel2LULC_354_augmented"

# Read train images and apply some transformations
train_datagenerator = ImageDataGenerator(rescale=1/255, width_shift_range=.15, height_shift_range=.15)
train_it = train_datagenerator.flow_from_directory(TRAIN_DIR, batch_size=10266)
# Iterate over the transformed images to save them along with the original ones
for file in train_it.filenames:
    img = load_img(TRAIN_DIR+file)  
    x = img_to_array(img) 
    x = x.reshape((1, ) + x.shape)  
    aug_it = train_datagenerator.flow(x, batch_size = 1, 
                      save_to_dir = AUG_TRAIN_DIR,  
                      save_prefix =file+"_shifted", save_format ='jpg')
    aug_it.next()
