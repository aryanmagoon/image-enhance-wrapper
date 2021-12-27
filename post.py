import requests
import numpy as np
import matplotlib.pyplot as plt
import json
from PIL import Image

files={'image': open('download.png', 'rb')}
url='http://localhost:5000/upsample'
print('request sent')
r=requests.post(url, files=files)
y=np.array(json.loads(r.text))
im = Image.fromarray((y * 255).astype(np.uint8))
im.show()