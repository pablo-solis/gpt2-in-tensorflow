import tensorflow as tf
import einops
from dataclasses import dataclass

# This class is used to initialize each layer


@dataclass
class Config:
  d_model: int = 768 # size of the residual stream
  debug: bool = True
  layer_norm_eps: float = 1e-5
  d_vocab: int = 50257

  # range of values upon initalization e.g. init in N(0, std=init_range)
  init_range: float = 0.02 

  # for positional embedding and size of context
  n_ctx: int = 1024 

  # dimension of attention head space
  d_head: int = 64 

  # dimension of mlp space
  d_mlp: int = 3072 
  n_heads: int = 12
  n_layers: int = 12

class LayerNorm(tf.keras.layers.Layer):
  def __init__(self, cfg):
      super().__init__()
      self.cfg = cfg
      self.w = tf.Variable(tf.ones([self.cfg.d_model]), name="gamma")
      self.b = tf.Variable(tf.zeros([self.cfg.d_model]), name="beta")

  def call(self, x):
      # x: [batch, position, d_model]

      # Compute mean and variance along d_model (last) axis
      mean = tf.reduce_mean(x, axis=-1, keepdims=True)
      variance = tf.reduce_mean(tf.square(x - mean), axis=-1, keepdims=True)

      # Normalize, scale, and shift
      normalized = (x - mean) / tf.sqrt(variance + self.cfg.layer_norm_eps)
      return normalized * self.w + self.b


class Embed(tf.keras.layers.Layer):
  def __init__(self, cfg):
      super().__init__()

      # Extract relevant configuration parameters
      self.cfg = cfg

      # Create the embedding matrix
      self.W_E = self.add_weight(
          name="embedding_matrix",
          shape=(self.cfg.d_vocab, self.cfg.d_model),
          initializer=tf.keras.initializers.RandomNormal(stddev=self.cfg.init_range),
          trainable=True  # Ensure the matrix is trainable
      )

  def call(self, tokens):
      # tokens: [batch, position]
      # TODO: explain
      embed = tf.nn.embedding_lookup(self.W_E, tokens)
      return embed
  

class PosEmbed(tf.keras.layers.Layer):
    def __init__(self, cfg):
      super().__init__()
      self.cfg = cfg

      # Create position embedding matrix
      self.W_pos = self.add_weight(
          name="position_embedding_matrix",
          shape=(self.cfg.n_ctx, self.cfg.d_model),
          initializer=tf.keras.initializers.RandomNormal(
                                            stddev=self.cfg.init_range),
          trainable=True
      )

    def call(self, tokens, verbose=False):
      # tokens has shape [batch, position]

      # Get batch size dynamically
      batch_size = tf.shape(tokens)[0].numpy()

      # Create position indices for each token
      position_indices = tf.range(tf.shape(tokens)[1])

      # shape [position, d_model]
      pos_embed = tf.gather(self.W_pos, position_indices)
      pos_embed = einops.repeat(pos_embed,
              "position d_model -> batch position d_model", batch=batch_size)
      return pos_embed
    
class Attention(tf.keras.layers.Layer):
  def __init__(self, cfg):
    super().__init__()
    self.cfg = cfg

    # Queries
    self.W_Q = self.add_weight(
        name="W_Q", shape=(cfg.n_heads, cfg.d_model, cfg.d_head),
        initializer=tf.keras.initializers.RandomNormal(stddev=cfg.init_range))
    self.b_Q = self.add_weight(
        name="b_Q", shape=(cfg.n_heads, cfg.d_head), initializer='zeros')

    # Keys
    self.W_K = self.add_weight(
        name="W_K", shape=(cfg.n_heads, cfg.d_model, cfg.d_head),
        initializer=tf.keras.initializers.RandomNormal(stddev=cfg.init_range))
    self.b_K = self.add_weight(
        name="b_K", shape=(cfg.n_heads, cfg.d_head), initializer='zeros')

    # Values
    self.W_V = self.add_weight(
        name="W_V", shape=(cfg.n_heads, cfg.d_model, cfg.d_head),
        initializer=tf.keras.initializers.RandomNormal(stddev=cfg.init_range))
    self.b_V = self.add_weight(
        name="b_V", shape=(cfg.n_heads, cfg.d_head), initializer='zeros')

    # Outputs
    self.W_O = self.add_weight(
        name="W_O", shape=(cfg.n_heads, cfg.d_head, cfg.d_model),
        initializer=tf.keras.initializers.RandomNormal(stddev=cfg.init_range))
    self.b_O = self.add_weight(
        name="b_O", shape=(cfg.d_model), initializer='zeros')

    # Ignore value for masking (large negative value)
    self.IGNORE = tf.constant(-1e5, dtype=tf.float32)
  def call(self, normalized_resid_pre):
    # normalized_resid_pre.shape = (batch, position, d_model) = bpd
    # W.shape = (n_heads, d_model, d_head) = ndh
    # b.shape = (n_heads, d_head) = nh

    # step1: dot residual stream with queries, keys, values
    # note tf.einsum supports compressed format bpd vs b p d
    # but doesn't support names with more than one charater
    # self.b_Q has shape nh but will be broadcast to shape bpnh
    q = tf.einsum("bpd, ndh ->bpnh", normalized_resid_pre, self.W_Q) + self.b_Q
    k = tf.einsum("bpd,ndh->bpnh", normalized_resid_pre, self.W_K) + self.b_K
    v = tf.einsum("bpd,ndh->bpnh", normalized_resid_pre, self.W_V) + self.b_V

    #step2: calculate attention scores
    # attention scores shape: batch head number query_pos key_pos
    # rename query_pos in q as q, rename key_pos in k as k
    # sum along h dimension
    attn_scores1 = tf.einsum("bqnh, bknh-> bnqk",q,k)

    # if we assuem q,k have entires that are ~N(0,1) then
    # then q@k/sqrt(d_head) roughly N(0,1)
    # this is variance reduction technique to help with numerical stability
    attn_scores1 = attn_scores1 / tf.sqrt(tf.cast(self.cfg.d_head, tf.float32))
    attn_scores = self.apply_causal_mask(attn_scores1)

    # softmax for prob dist https://youtu.be/eMlx5fFNoYc?si=ob4SBcpEYgC-M35H&t=564
    attn = tf.nn.softmax(attn_scores, axis=-1)

    # step3: weighted sum of attn and values
    # sum over key position
    # z is the weighted sum of values by how much attention to pay to them
    # information from the key position is brought forward to the q position
    z = tf.einsum("bnqk,bknh->bqnh", attn, v)

    # finally map back from d_head dimensions to d_model dimensions
    # sum over n,h
    attn_out = tf.einsum("bpnh,nhd->bpd", z, self.W_O) + self.b_O

    return attn_out

  def apply_causal_mask(self, attn_scores):
    # make attention scores *strictly* upper triangular
    # where does this convention come from?

    #  tf.linalg.band_part takes input and sets some values to 0
    # key special cases:
    # tf.linalg.band_part(input, 0, -1) ==> Upper triangular part.
    # tf.linalg.band_part(input, -1, 0) ==> Lower triangular part.
    # tf.linalg.band_part(input, 0, 0) ==> Diagonal.

    # rows -> query_pos
    # cols -> key_pos
    # causal means query_pos > key_pos
    # non casual means query_pos < key_pos mean upper triangular entires

    # lower triangular mask:
    mask = tf.linalg.band_part(tf.ones_like(attn_scores), -1, 0)

    # has False *strictly* above diagonal and True on diagonal and below
    mask = tf.cast(mask, tf.bool)

    # has True *strictly* above diagonal
    #this also works: mask = ~mask
    mask = tf.logical_not(mask)
    

    # replace lower diagonal entries with self.IGNORE
    attn_scores = tf.where(mask, self.IGNORE, attn_scores)
    return attn_scores
  
class MLP(tf.keras.layers.Layer):
  def __init__(self, cfg):
    super(MLP, self).__init__()
    self.cfg = cfg
    self.init_range = cfg.init_range

    # Weights and Biases
    self.W_in = self.add_weight(name="W_in", shape=(cfg.d_model, cfg.d_mlp), 
                                initializer="random_normal", trainable=True)
    self.b_in = self.add_weight(name="b_in", shape=(cfg.d_mlp,),
                                initializer="zeros", trainable=True)
    self.W_out = self.add_weight(name="W_out", shape=(cfg.d_mlp, cfg.d_model),
                                 initializer="random_normal", trainable=True)
    self.b_out = self.add_weight(name="b_out", shape=(cfg.d_model,),
                                 initializer="zeros", trainable=True)

  def call(self, normalized_resid_mid):
    # Sum over d_model dimension
    pre = tf.einsum("bpd,dm->bpm", normalized_resid_mid, self.W_in) + self.b_in

    # GELU Activation (TensorFlow's built-in)
    post = tf.nn.gelu(pre)

    mlp_out = tf.einsum("bpm,md->bpd", post, self.W_out) + self.b_out
    return mlp_out

class TransformerBlock(tf.keras.layers.Layer):
  def __init__(self, cfg):
    super().__init__()
    self.cfg = cfg
    self.ln1 = LayerNorm(cfg)
    self.attn = Attention(cfg)
    self.ln2 = LayerNorm(cfg)
    self.mlp = MLP(cfg)

  def call(self, resid_pre):
    normalized_resid_pre = self.ln1(resid_pre)
    attn_out = self.attn(normalized_resid_pre)

    # this step was original reason why
    # you would want n_heads*d_head = d_model
    resid_mid = resid_pre + attn_out
    normalized_resid_mid = self.ln2(resid_mid)
    mlp_out = self.mlp(normalized_resid_mid)
    resid_post = resid_mid + mlp_out
    return resid_post
  
class Unembed(tf.keras.layers.Layer):
  def __init__(self, cfg):
    super().__init__()
    self.cfg = cfg
    self.init_range = cfg.init_range

    # Weights and Biases
    self.W_U = self.add_weight(
        name="W_U", shape=(cfg.d_model, cfg.d_vocab), initializer="random_normal", trainable=True
    )
    self.b_U = self.add_weight(
        name="b_U", shape=(cfg.d_vocab,), initializer="zeros", trainable=False  # Set trainable to False
    )

  def call(self, normalized_resid_final):
    logits = tf.einsum("bpd,dv->bpv", normalized_resid_final, self.W_U) + self.b_U
    return logits
  
class GPT2(tf.keras.Model):
  def __init__(self, cfg):
    super().__init__()
    self.cfg = cfg
    self.embed = Embed(cfg)
    self.pos_embed = PosEmbed(cfg)
    self.blocks = [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
    self.ln_final = LayerNorm(cfg)
    self.unembed = Unembed(cfg)

  def call(self, x):
    embed = self.embed(x)
    pos_embed = self.pos_embed(x)
    resid = embed + pos_embed

    for block in self.blocks:
      resid = block(resid)
    
    normalized_resid_final = self.ln_final(resid)
    logits = self.unembed(normalized_resid_final)
    return logits