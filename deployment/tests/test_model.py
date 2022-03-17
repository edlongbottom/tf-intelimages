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
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
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

# test output format for prediction
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
def test_predict_proba():
    for model_path in paths:
        clf = models.load_model(model_path)
        y_pred = clf.predict(X_test)
        assert np.max(y_pred) > 0.5, f"All prediction probabilities under 50% for model: {model_path}"

# test model accuracy greater than 75% on test set
"""
DATA_DIR = 'C:\\Users\\eddlo\\Python\\Projects\\TF-images\\data'
test_data_dir = os.path.join(DATA_DIR,'seg_test','seg_test')
def test_model_acc():
    # retrieve test data samples
    test_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
        directory=test_data_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical')
    X_test, y_test = next(test_generator)

    # run each model on the test dataset and check whether accuracy is above a threshold (65%)
    for model_path in paths:
        clf = models.load_model(model_path)
        y_pred = clf.predict(X_test)
        loss, acc = clf.evaluate(x=X_test,y=y_test)
        assert acc >= 0.65, f"Accuracy less than 65% for model: {model_path}"
"""