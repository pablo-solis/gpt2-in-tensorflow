import tensorflow as tf
import numpy as np
import torch 


def torch_gpt2_test(cls: tf.keras.layers.Layer, tf_input: tf.Tensor, 
                    reference_gpt2: torch.nn.Module, layer: str) -> float:
  """
  take tf layer and pytorch layer with same architechture
  set weights of tf layer to match the pytorch weights
  compare results on tf_input to see if they agree.
  """
  param_names = {
    "embed": ['W_E'],
    "pos_embed": ['W_pos'],
    "blocks.0.ln1": ['w','b'],
    "blocks.0.attn": ['W_Q','b_Q','W_K','b_K','W_V','b_V','W_O','b_O'],
    "blocks.0.mlp": ['W_in','b_in','W_out','b_out'],
    "unembed": ['W_U','b_U']
    }
  
  # parse layer and get torch version
  lst = layer.split('.')
  if len(lst)==1:
    torch_layer = getattr(reference_gpt2, lst[0])
  elif lst[0]=='blocks':
    torch_layer = getattr(reference_gpt2.blocks[0],lst[2])

  # tf version
  tf_layer = cls(cfg)

  # set weights from torch layer
  for name in param_names[layer]:

    if hasattr(torch_layer, name) and hasattr(tf_layer, name):
      tf_param = getattr(tf_layer, name)
      new_params = tf.constant(getattr(torch_layer, name).detach().numpy())
      tf_param.assign(new_params)
    else:
      raise AttributeError(f"{name} not found in torch_layer or pytorch_layer")

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