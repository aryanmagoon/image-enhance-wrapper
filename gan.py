import cv2
import math
import numpy as np
import torch
from torch.nn import functional
import os
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
class EnhanceAgent():
  def __init__(self, scale, model_path,  model=None, tile=0, tile_pad=10, pre_pad=10, half=False):
    self.scale=scale
    self.tile_size=tile
    self.tile_pad = tile_pad
    self.pre_pad=pre_pad
    self.half=half
    self.mod_scale=None

    self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loadnet=torch.load('ImageEnhance.pth', map_location=torch.device('cpu'))
    if 'params_ema' in loadnet:
      keyname='params_ema'
    else:
      keyname='params'
    model.load_state_dict(loadnet[keyname], strict=True)
    model.eval()
    self.model= model.to(self.device)
    if self.half:
      self.model=self.model.half()
  def pre_process(self, image):
    image=torch.from_numpy(np.transpose(image, (2,0,1))).float()
    self.image= image.unsqueeze(0).to(self.device)
    if(self.half):
      self.image=self.image.half()
    if self.pre_pad>0:
      self.image=functional.pad(self.image, (0, self.pre_pad, 0, self.pre_pad), 'reflect')
    if self.scale==2:
      self.mod_scale=2
    elif self.scale==1:
      self.mod_scale=4
    if self.mod_scale is not None:
      self.mod_pad_h, self.mod_pad_w=0,0
      _,_,h,w=self.image.size()
      if (h % self.mod_scale !=0):
        self.mod_pad_h = (self.mod_scale-h % self.mod_scale)
      if (w % self.mod_scale !=0):
        self.mod_pad_w = (self.mod_scale-h%self.mod_scale)
      self.image = functional.pad(self.image, (0, self.mod_pad_w, 0, self.mod_pad_h), 'reflect')
  def process(self):
    self.output=self.model(self.image)
  def post_process(self):
    if(self.mod_scale is not None):
      _,_, h, w=self.output.size()
      self.output=self.output[:,:, 0:h-self.mod_pad_h * self.scale, 0:w-self.mod_pad_w*self.scale]
    if self.pre_pad !=0:
      _,_, h, w=self.output.size()
      self.output=self.output[:,:, 0:h-self.pre_pad * self.scale, 0:w-self.pre_pad*self.scale]
    return self.output
    
  @torch.no_grad()
  def enhance(self, image, outscale=None, alpha_upsampler='realesrgan'):
    h_input, w_input = image.shape[0:2]
    image=image.astype(np.float32)
    if np.max(image)>256:
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
    output=output.data.squeeze().float().cpu().clamp_(0,1).numpy()
    output=np.transpose(output[[2,1,0],:,:], (1,2,0))
    if img_mode=='L':
      output=cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    if img_mode=='RGBA':
      if(alpha_upsampler=='realesrgan'):
        self.pre_process(alpha)
        self.process()
        output_alpha=self.post_process()
        output_alpha=output_alpha.data.squeeze().float().cpu().clamp_(0,1).numpy()
        output_alpha = np.transpose(output_alpha[[2,1,0],:,:], (1,2,0))
        output_alpha = cv2.cvtColor(output_alpha, cv2.COLOR_BGR2GRAY)
      else:
          h,w=alpha.shape[0:2]
          output_alpha = cv2.resize(alpha, (w* self.scale, h*self.scale), interpolation=cv2.INTER_LINEAR)
      output=cv2.cvtColor(output, cv2.COLOR_BGR2BGRA)
      output[:,:,3]=output_alpha
    if max_range==65535:
      final=(output*65535.0).round().astype(np.uint16)
    else:
      final=(output*255.0).round().astype(np.uint8)
    if outscale is not None and outscale !=float(self.scale):
      final=cv2.resize(
          output, (int(w_input*outscale), int(h_input*outscale), ), interpolation=cv2.INTER_LANCZOS4)
    return final
