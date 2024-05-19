import tensorflow as tf
import numpy as np
import torch 

from .layers import Config
from typing import Type

cfg = Config()

def set_weights_from_torch(tf_layer: tf.keras.layers.Layer,
                           torch_layer: torch.nn.Module)-> tf.keras.layers.Layer:
  """
  This will take named parameters in torch_layer and assign them to tf_layer
  e.g. 
  - tf_layer.Weight = torch_layer.Weight
  - tf_layer.bias = torch_layer.bias
  """
  for name, tensor in torch_layer.named_parameters():
    if hasattr(tf_layer, name):
      tf_param = getattr(tf_layer,name)
      new_param = tf.constant(tensor.detach().numpy())
      tf_param.assing(new_param)
    else: 
      raise AttributeError(f"tf_layer doesn't have attribute {name}")
  return tf_layer

def torch_gpt2_test(cls: Type[tf.keras.layers.Layer], tf_input: tf.Tensor, 
                    reference_gpt2: torch.nn.Module, layer: str) -> float:
  """
  take tf layer and pytorch layer with same architechture
  set weights of tf layer to match the pytorch weights
  compare results on tf_input to see if they agree.
  """

  # parse layer and get torch version
  lst = layer.split('.')
  if len(lst)==1:
    torch_layer = getattr(reference_gpt2, lst[0])
  elif lst[0]=='blocks':
    torch_layer = getattr(reference_gpt2.blocks[0],lst[2])

  # tf version
  tf_layer = cls(cfg)
  tf_layer = set_weights_from_torch(tf_layer, torch_layer)
  correct_values = compare_tf_pytorch(tf_layer, torch_layer, tf_input)

  return correct_values

def compare_tf_pytorch(tf_layer, pytorch_layer, tf_input):
  """
  compares if ouput of tf layer and pytorch layer on same input
  gives the same numpy array
  """
  tf_output = tf_layer(tf_input)
  torch_output = pytorch_layer(torch.tensor(tf_input.numpy()))

  comp = tf.experimental.numpy.isclose(
      tf_output.numpy(), torch_output.detach().numpy(), rtol=1e-2, atol=1e-2)
  correct_values = (comp.numpy().sum() / tf.size(comp)).numpy()
  
  return f"correct pct: {100*correct_values:.2f}"