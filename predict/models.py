
# Extract features using NASNetLarge as extractor.
from keras.applications.nasnet import NASNetLarge, preprocess_input
nasnet_preprocessor = preprocess_input

# Extract features using EfficientNetB7 as extractor.
from tensorflow.keras.applications.efficientnet import EfficientNetB7, preprocess_input
efficientnet_b7_preprocessor = preprocess_input
