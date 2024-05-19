# gpt2-in-tensorflow

Based on this [colab](https://colab.sandbox.google.com/github/neelnanda-io/Easy-Transformer/blob/clean-transformer-demo/Clean_Transformer_Demo.ipynb) from Neel Nanda

Run a demo in this [colab](https://colab.research.google.com/drive/1p7mvtIPaJ6sI2nQGzgaIW67Vq2FnmWcF?authuser=1#scrollTo=_L79Numv5dCS).

GPT-2 in TensorFlow: A Layer-by-Layer Implementation

This repository provides a detailed, from-scratch implementation of the GPT-2 transformer architecture in TensorFlow. It's designed for educational purposes, breaking down the model into individual layers for clear understanding.

Features

- Implements essential GPT-2 components:
  - Embed: Input token embedding layer
  - PosEmbed: Positional embedding layer
  - Attention: Multi-head self-attention mechanism
  - LayerNorm: Layer normalization
  - MLP: Multi-layer perceptron (feedforward) layer
  - TransformerBlock: Combines multiple layers into a reusable block
  - Unembed: Output layer for generating logits
- Weight Transfer from PyTorch: Includes functionality to seamlessly transfer pre-trained weights from a PyTorch GPT-2 model to the corresponding TensorFlow implementation.

- Testing Framework: Provides a testing mechanism (torch_gpt2_test) to verify the correctness of the TensorFlow implementation by comparing its output with a reference PyTorch GPT-2 model.


Project Goals

- Educational Resource: Help learners understand the inner workings of GPT-2 by building it piece-by-piece.
- TensorFlow Expertise: Enhance your TensorFlow skills through practical model development.
- Research and Adaptation: Serve as a foundation for further research and customization of GPT-2 in TensorFlow.