from .preprocess import preprocess
import numpy as np
import keras
from .config import img_size
from .get_features import get_features
from .models import NASNetLarge, nasnet_preprocessor, EfficientNetB7, efficientnet_b7_preprocessor


def predict(img):
    X = preprocess(img, img_size)
    nasnet_features = get_features(NASNetLarge,
                                   nasnet_preprocessor,
                                   img_size, [X])
    efficientnet_b7_features = get_features(EfficientNetB7,
                                            efficientnet_b7_preprocessor,
                                            img_size, [X])
    features = np.concatenate([nasnet_features,
                               efficientnet_b7_features], axis=-1)
    dnn = keras.models.load_model('./models/DogBreedModel.h5')
    y_pred = dnn.predict(features)
    return y_pred
