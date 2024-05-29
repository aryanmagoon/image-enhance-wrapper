from gan_flax import EnhanceAgent_JAX
import flask
from PIL import Image
import io
import numpy as np
import json


app=flask.Flask(__name__)

upscale=EnhanceAgent_JAX(scale=4, model_params="jax_weights_final.pickle", pre_pad=10)


@app.route("/upsample", methods=["POST"])
def upsample():
    global upscale
    if flask.request.method =="POST":
        if flask.request.files.get("image"):
            print('got request')
            image=Image.open(io.BytesIO(flask.request.files["image"].read()))
            if('scale' in flask.request.form):
                scale=int(flask.request.form['scale'])
            print(scale)
            print(type(scale))
            image=np.array(image)
            final=upscale.enhance(image, outscale=scale)
            jsonfinal=json.dumps(final.tolist())
            return jsonfinal
        
@app.route("/productdev", methods=["POST"])
def upsampling():
    global upscale
    if flask.request.method =="POST":
        if 'img' in flask.request.form:
            image_list = json.loads(flask.request.form['img'])
            # Convert the list to a numpy array
            image = np.array(image_list)            
            print(image.shape)
        if('scale' in flask.request.form):
            scale=float(flask.request.form['scale'])
        print(scale)
        print(type(scale))
        final=upscale.enhance(image, outscale=scale)
        jsonfinal=json.dumps(final.tolist())
        return jsonfinal






