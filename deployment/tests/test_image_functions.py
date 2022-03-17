"""
Test image functions

Unit tests for testing the functions in image_functions.py

"""

# Python will only search current directory and installed packages
# To find a module that isn't installed, add it to sys path 
import os, sys
import numpy as np

BASE_DIR = 'C:\\Users\\eddlo\\Python\\Projects\\TF-images\\TF-IntelImages\\deployment'
sys.path.append(BASE_DIR)

from app.image_functions import get_img, pull_imgs, convert_from_bytes, convert_from_jpeg

with open(os.path.join(BASE_DIR,'app','pexels_api.txt'),'r') as f:
    api_key = f.read()



def test_get_img(category="sea"):
    output = get_img(category, api_key, base_url="http://api.pexels.com/v1/")
    assert isinstance(output, bytes)

def test_pull_imgs(category="glacier", num_imgs=1):
    img_dir = os.path.join(BASE_DIR,'app','images')
    pull_imgs(category, api_key, img_dir, num_imgs, base_url="http://api.pexels.com/v1/")
    assert os.path.exists(os.path.join(img_dir, category))
    assert len(os.listdir(os.path.join(img_dir, category))) == num_imgs
    for img in os.listdir(os.path.join(img_dir,category)):
        os.remove(os.path.join(img_dir,category,img))

def test_convert_from_jpeg():
    img_path = os.path.join(BASE_DIR,'tests','img_for_testing.jpg')
    arr = convert_from_jpeg(img_path, size=(150,150))
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (150, 150, 3)

def test_convert_from_bytes():
    img_bytes_path = os.path.join(BASE_DIR,'tests','img_bytes')
    with open(img_bytes_path,'rb') as f:
        img_bytes = f.read()
    arr = convert_from_bytes(img_bytes, size=(150,150))
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (150, 150, 3)
