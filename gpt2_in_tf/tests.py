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

  # use _tf_layer variable to track current tf_layer
  # don't overwrite tf_layer variable
  for full_name, tensor in torch_layer.named_parameters():
    # handle paramers named e.g. blocks.0.mlp.W_out
    if full_name.startswith("blocks.") and len(full_name.split('.'))==4:
      _,idx,layer,name = full_name.split('.')
      _tf_layer = getattr(tf_layer.blocks[int(idx)],layer)

    # handle paramters named mlp.b_out
    elif len(full_name.split('.'))==2:
      layer, name = full_name.split('.')
      _tf_layer = getattr(tf_layer,layer)
    
    # handle paramters named b_out
    elif len(full_name.split('.'))==1:
      name = full_name
      _tf_layer=tf_layer

    # reassign tf param
    if hasattr(_tf_layer, name):
      tf_param = getattr(_tf_layer,name)
      
      # when on GPU move to CPU before converting to NumPy 
      if tensor.is_cuda:
        tensor = tensor.cpu()
      new_param = tf.constant(tensor.detach().numpy())
      tf_param.assign(new_param)
    else: 
      raise AttributeError(f"tf_layer of type({type(_tf_layer)}) doesn't have attribute {name}")
  return tf_layer

def torch_gpt2_test(cls: Type[tf.keras.layers.Layer], torch_layer: torch.nn.Module,
                    tf_input: tf.Tensor,) -> float:
  """
  take tf layer and pytorch layer with same architechture
  set weights of tf layer to match the pytorch weights
  compare results on tf_input to see if they agree.
  """
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

  if torch_output.is_cuda:
    torch_output = torch_output.cpu()
  
  comp = tf.experimental.numpy.isclose(
      tf_output.numpy(), torch_output.detach().numpy(), rtol=1e-2, atol=1e-2)
  correct_values = (comp.numpy().sum() / tf.size(comp)).numpy()
  
  return f"correct pct: {100*correct_values:.2f}"