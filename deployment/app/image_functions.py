"""
Image Functions

Various functions for downloading images and conversion to numpy arrays 

"""
import io
import os
import random

import numpy as np
import requests
from PIL import Image


def get_img(category, api_key, base_url="http://api.pexels.com/v1/"):
    '''Queries Pexels API for a single image and returns it in byte format'''
    r_search = requests.get(url=base_url+f"search?query={category}", headers={"Authorization": api_key})
    i = random.randrange(0,len(r_search.json()["photos"]),1)    
    img_id = r_search.json()["photos"][i]['id']

    r_photo = requests.get(url=base_url+f"photos/{img_id}", headers={"Authorization": api_key})
    img_url = r_photo.json()['src']['tiny']

    r_img = requests.get(url=img_url, stream=True)
    img = r_img.raw.read()

    return img


def pull_imgs(category, api_key, img_dir, num_imgs=1, base_url="http://api.pexels.com/v1/"):
    '''Queries Pexels API for images and saves them locally as per arguments'''
    r_search = requests.get(url=base_url+f"search?query={category}", headers={"Authorization": api_key})
    for i in range(num_imgs):
        if i < len(r_search.json()["photos"]):
            img_id = r_search.json()["photos"][i]['id']

            r_photo = requests.get(url=base_url+f"photos/{img_id}", headers={"Authorization": api_key})
            img_url = r_photo.json()['src']['tiny']

            r_img = requests.get(url=img_url, stream=True)
            img = r_img.raw.read()
            file_path = os.path.join(img_dir, category, f"{img_id}.jpg")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path,"wb") as f:
                f.write(img)


def convert_from_jpeg(img_path, size=(150,150)):
    '''Resizes an image from path and converts into a numpy array with format "chanels last", e.g. (150,150,3)'''
    img = Image.open(img_path)
    img_resized = img.resize(size=size)
    return np.asarray(img_resized)


def convert_from_bytes(img_bytes, size=(150,150)):
    '''Resizes an image from bytes and converts into a numpy array with format "chanels last", e.g. (150,150,3)'''
    img = Image.open(io.BytesIO(img_bytes))
    img_resized = img.resize(size=size)
    return np.asarray(img_resized)
