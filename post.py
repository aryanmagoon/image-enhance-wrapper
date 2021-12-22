import requests
import numpy as np
import matplotlib.pyplot as plt
import json

files={'image': open('download.png', 'rb')}
url='http://127.0.0.1:5000/upsample'

r=requests.post(url, files=files)
y=np.array(json.loads(r.text))
plt.imshow(y)
plt.show()