from typing import Sequence
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn

def pixel_unshuffle(x, scale):
    b, h, w, c = x.shape  # Assuming NHWC format
    new_h = h // scale
    new_w = w // scale
    x = x.reshape(b, new_h, scale, new_w, scale, c)
    x = jnp.transpose(x, (0, 1, 3, 2, 4, 5))
    x = x.reshape(b, new_h, new_w, c * (scale ** 2))
    return x


def make_layer(basic_block, num_basic_block, **kwarg):
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(layers)

class ResidualDenseBlock(nn.Module):
  num_feat : int
  num_grow_ch : int
  def setup(self):
    self.conv1 = nn.Conv(self.num_grow_ch, kernel_size=(3,3), strides = (1,1), padding =((1,1),(1,1)))
    self.conv2 = nn.Conv(self.num_grow_ch, kernel_size=(3,3), strides = (1,1), padding =((1,1),(1,1)))
    self.conv3 = nn.Conv(self.num_grow_ch, kernel_size=(3,3), strides = (1,1), padding =((1,1),(1,1)))
    self.conv4 = nn.Conv(self.num_grow_ch, kernel_size=(3,3), strides = (1,1), padding =((1,1),(1,1)))
    self.conv5 = nn.Conv(self.num_feat, kernel_size=(3,3), strides = (1,1), padding =((1,1),(1,1)))

  @nn.compact
  def __call__(self, x):
    x1 = nn.activation.leaky_relu(self.conv1(x), negative_slope=0.2)
    x2 = nn.activation.leaky_relu(self.conv2(jnp.concatenate((x, x1), -1)), negative_slope=0.2)
    x3 = nn.activation.leaky_relu(self.conv3(jnp.concatenate((x, x1, x2), -1)), negative_slope=0.2)
    x4 = nn.activation.leaky_relu(self.conv4(jnp.concatenate((x, x1, x2, x3), -1)), negative_slope=0.2)
    x5 = self.conv5(jnp.concatenate((x, x1, x2, x3, x4), -1))
    return x5 * 0.2 + x
  

class RRDB(nn.Module):
  num_feat : int
  num_grow_ch : int
  def setup(self):
    self.rdb1 = ResidualDenseBlock(self.num_feat, self.num_grow_ch)
    self.rdb2 = ResidualDenseBlock(self.num_feat, self.num_grow_ch)
    self.rdb3 = ResidualDenseBlock(self.num_feat, self.num_grow_ch)

  @nn.compact
  def __call__(self, x):
    out = self.rdb1(x)
    out = self.rdb2(out)
    out = self.rdb3(out)
    return out * 0.2 + x
  
class RRDBNet_Flax(nn.Module):
    num_in_ch : int
    num_out_ch : int
    scale : int
    num_feat : int
    num_blocks : int
    num_grow_ch : int

    def setup(self):
        if self.scale == 2:
            self.num_in_ch = self.num_in_ch * 4
        elif self.scale == 1:
            self.num_in_ch = self.num_in_ch * 16
        self.conv_first = nn.Conv(self.num_feat, kernel_size=(3,3), strides = (1,1), padding =((1,1),(1,1)))

        self.body = make_layer(RRDB, self.num_blocks, num_feat=self.num_feat, num_grow_ch=self.num_grow_ch)
        self.conv_body = nn.Conv(self.num_feat, kernel_size=(3,3), strides = (1,1), padding =((1,1),(1,1)))

        self.conv_up1 = nn.Conv(self.num_feat, kernel_size=(3,3), strides = (1,1), padding =((1,1),(1,1)))
        self.conv_up2 = nn.Conv(self.num_feat, kernel_size=(3,3), strides = (1,1), padding =((1,1),(1,1)))
        self.conv_hr = nn.Conv(self.num_feat, kernel_size=(3,3), strides = (1,1), padding =((1,1),(1,1)))
        self.conv_last = nn.Conv(self.num_out_ch, kernel_size=(3,3), strides = (1,1), padding =((1,1),(1,1)))
    
    @nn.compact
    def __call__(self, x):
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = nn.activation.leaky_relu((self.conv_up1(jax.image.resize(feat, (feat.shape[0], feat.shape[1] * 2, feat.shape[2] * 2, feat.shape[3]), method='nearest'))), negative_slope = 0.2)
        feat = nn.activation.leaky_relu((self.conv_up2(jax.image.resize(feat, (feat.shape[0], feat.shape[1] * 2, feat.shape[2] * 2, feat.shape[3]), method='nearest'))), negative_slope = 0.2)
        out = self.conv_last(nn.activation.leaky_relu((self.conv_hr(feat)), negative_slope = 0.2))
        return out