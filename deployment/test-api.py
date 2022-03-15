import requests
import json

from app.image_functions import get_img, convert_from_bytes

# retrieve API key
with open('app/pexels_api.txt','r') as f:
    api_key = f.read()

# retrieve a random image from the pexels API and convert to a numpy array 
img_bytes = get_img("forest", api_key, base_url="http://api.pexels.com/v1/")
arr = convert_from_bytes(img_bytes, size=(150,150))

# define inputs for API request
host, port = "127.0.0.1", "54599"

url = f"http://{host}:{port}/v1/models/lenet5:predict"
headers = {'Content-Type': 'application/json'}
data = json.dumps({"instances":[arr.tolist()]})

# send a request and process output
r = requests.post(url=url, headers=headers, data=data)
if r.status_code == 200: predict_proba = r.json()['predictions']

