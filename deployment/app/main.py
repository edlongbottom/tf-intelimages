from image_functions import convert_from_bytes, get_img

# read API key
with open('pexels_api.txt','r') as f:
    api_key = f.read()

# pull an image and convert to a numpy array
img_bytes = get_img('street', api_key=api_key)
img_arr = convert_from_bytes(img_bytes)

# pass 
