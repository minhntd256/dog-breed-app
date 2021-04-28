import numpy as np
from tqdm import tqdm
import tensorflow as tf

def preprocess(img, img_size = (224,224,3)):
  X = np.zeros([1, img_size[0], img_size[1], 3], dtype=np.uint8)
  img_pixels = tf.keras.preprocessing.image.load_img(img, target_size=img_size)
  X[0] = img_pixels
  return X