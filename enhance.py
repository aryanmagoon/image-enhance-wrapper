from gan import EnhanceAgent
from RRDBNet import RRDBNet
import flask
from PIL import Image
import io
import numpy as np
import cv2
import json


app=flask.Flask(__name__)

upscale=EnhanceAgent(scale=4, model_path="ImageEnhance.pth", model=RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4), tile=0,)


@app.route("/upsample", methods=["POST"])
def upsample():
    global upscale
    if flask.request.method =="POST":
        if flask.request.files.get("image"):
            image=Image.open(io.BytesIO(flask.request.files["image"].read()))
            image=np.array(image)
            final=upscale.enhance(image, outscale=4)
            jsonfinal=json.dumps(final.tolist())
            return jsonfinal




