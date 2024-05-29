import cv2
import numpy as np
import jax.numpy as jnp
from RRDBNet_Flax import RRDBNet_Flax
import pickle
import numpy as np

class EnhanceAgent_JAX():
  def __init__(self, scale, model=None, model_params=None, pre_pad=10):
    self.scale=scale
    self.pre_pad=pre_pad
    self.mod_scale=None
    self.model_params = model_params
    self.model = model
    if self.model == None:
      self.model=RRDBNet_Flax(num_in_ch=3, num_out_ch=3, num_feat=64, num_blocks=23, num_grow_ch=32, scale=4)
    else:
      self.model=model
    #open jax_weights_final.pickle
    with open(self.model_params, 'rb') as handle:
        self.params = pickle.load(handle)
  def pre_process(self, image):
    image=jnp.array(image).astype(jnp.float32)
    self.image = jnp.expand_dims(image, axis=0)
    if self.pre_pad>0:
      self.image = jnp.pad(self.image, ((0, 0), (0, self.pre_pad), (0, self.pre_pad), (0, 0)), mode='reflect')
    if self.scale==2:
      self.mod_scale=2
    elif self.scale==1:
      self.mod_scale=4
    if self.mod_scale is not None:
      self.mod_pad_h, self.mod_pad_w=0,0
      _,h,w,_ =self.image.size()
      if (h % self.mod_scale !=0):
        self.mod_pad_h = (self.mod_scale-h % self.mod_scale)
      if (w % self.mod_scale !=0):
        self.mod_pad_w = (self.mod_scale-h%self.mod_scale)
      self.image = jnp.pad(self.image, ((0, 0), (0, self.mod_pad_h), (0, self.mod_pad_w), (0, 0)), mode='reflect')
  def process(self):
    self.output=self.model.apply(self.params, self.image)
  def post_process(self):
    if(self.mod_scale is not None):
      _,h, w, _=self.output.shape
      self.output=self.output[:, 0:h-self.mod_pad_h * self.scale, 0:w-self.mod_pad_w*self.scale, :]
    if self.pre_pad !=0:
      _,h, w, _=self.output.shape
      self.output=self.output[:, 0:h-self.pre_pad * self.scale, 0:w-self.pre_pad*self.scale, :]
    return self.output
    
  def enhance(self, image, outscale=None, alpha_upsampler='realesrgan'):
    h_input, w_input = image.shape[0:2]
    image=image.astype(np.float32)
    if jnp.max(image)>256:
      max_range=65535
    else:
      max_range=255
    image=image/max_range
    if len(image.shape)==2:
      img_mode='L'
      image=cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2]==4:
      img_mode ='RGBA'
      alpha = image[:,:,3]
      image=image[:,:,0:3]
      image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      if alpha_upsampler=='realesrgan':
        alpha=cv2.cvtColor(alpha, cv2.COLOR_GRAY2RGB)
    else:
      img_mode ='RGB'
      image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    self.pre_process(image)
    self.process()
    print('done')
    output=self.post_process()
    output=output.squeeze().astype(jnp.float32).clip(0,1)
    output = output[:, :, [2, 1, 0]]
    if img_mode=='L':
      output=cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    if img_mode=='RGBA':
      if(alpha_upsampler=='realesrgan'):
        self.pre_process(alpha)
        self.process()
        output_alpha=self.post_process()
        output_alpha= output_alpha.squeeze().astype(jnp.float32).clip(0,1)
        output_alpha = output_alpha[:, :, [2, 1, 0]]
        output_alpha = cv2.cvtColor(output_alpha, cv2.COLOR_BGR2GRAY)
      else:
          h,w=alpha.shape[0:2]
          output_alpha = cv2.resize(alpha, (w* self.scale, h*self.scale), interpolation=cv2.INTER_LINEAR)
      output=cv2.cvtColor(output, cv2.COLOR_BGR2BGRA)
      output[:,:,3]=output_alpha
    if max_range==65535:
      final=(output*65535.0).round().astype(jnp.uint16)
    else:
      final=(output*255.0).round().astype(jnp.uint8)
    if outscale is not None and outscale !=float(self.scale):
      final=cv2.resize(
          np.array(output), (int(w_input*outscale), int(h_input*outscale), ), interpolation=cv2.INTER_LANCZOS4)
    return final