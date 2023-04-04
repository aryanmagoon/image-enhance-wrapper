import requests
import numpy as np
import matplotlib.pyplot as plt
import json
from PIL import Image

files={'image': open('download.png', 'rb')}
url='http://localhost:5000/upsample'
print('request sent')
#print the size of the image
print(Image.open('download.png').size)
r=requests.post(url, files=files)
y=np.array(json.loads(r.text))
print(y.shape)
im = Image.fromarray((y * 255).astype(np.uint8))
im.show()