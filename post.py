import requests
import numpy as np
import matplotlib.pyplot as plt
import json

files={'image': open('download.png', 'rb')}
url='http://172.17.0.2:5000/upsample'
print('request sent')
r=requests.post(url, files=files)
y=np.array(json.loads(r.text))
print('got requests')
plt.imshow(y)
plt.show