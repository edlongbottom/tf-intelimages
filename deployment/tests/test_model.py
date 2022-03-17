"""
Test models

Unit test for testing each of the TF models' performance sitting under the /models directory

"""
import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models

BASE_DIR = 'C:\\Users\\eddlo\\Python\\Projects\\TF-images\\TF-IntelImages\\deployment'
MODELS_DIR = os.path.join(BASE_DIR,'models')
sys.path.append(BASE_DIR)

from app.image_functions import convert_from_jpeg 

# generate list of models
paths = []
for model in os.listdir(MODELS_DIR):
    for model_version in os.listdir(os.path.join(MODELS_DIR,model)):
        paths.append(os.path.join(MODELS_DIR, model, model_version))

# generate some test data
img_path = os.path.join(BASE_DIR, 'tests', 'img_for_testing.jpg')
X_test = [convert_from_jpeg(img_path, size=(150,150)).tolist()]

# test output format
def test_output_format():
    for model_path in paths:
        clf = models.load_model(model_path)
        y_pred = clf.predict(X_test)
        assert isinstance(y_pred, np.ndarray), f"Incorrect output format for prediction on test image for model: {model_path}"
        assert y_pred.shape == (1,6), f"Incorrect output shape for prediction on test image for model: {model_path}"

# test correct answer for test image
def test_model_correct():
    for model_path in paths:
        clf = models.load_model(model_path)
        y_pred = clf.predict(X_test)
        assert np.array(y_pred).argmax(axis=-1)[0] == 1, f"Incorrect classification of test image for model: {model_path}"
        
# test prediction probability above some threshold (50%)
def test_model_accuracy():
    for model_path in paths:
        clf = models.load_model(model_path)
        y_pred = clf.predict(X_test)
        assert np.max(y_pred) > 0.5, f"All prediction probabilities under 50% for model: {model_path}"
