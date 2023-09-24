import numpy as np
import tensorflow as tf
from env import Hangman
import string
from keras.preprocessing.sequence import pad_sequences

class PositionalEmbedding(tf.keras.layers.Layer):
  def __init__(self, vocab_size, embed_size, maxlen):
    super().__init__()
    self.d_model = embed_size
    self.embedding = tf.keras.layers.Embedding(vocab_size, self.d_model)
    self.pos_encoding = self.positional_encoding(length=maxlen, depth=self.d_model)

  @staticmethod
  def positional_encoding(length, depth):
    depth = depth/2
    positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)
    angle_rates = 1 / (10000**depths)         # (1, depth)
    angle_rads = positions * angle_rates      # (pos, depth)
    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1)
    return tf.cast(pos_encoding, dtype=tf.float32)

  def compute_mask(self, *args, **kwargs):
    return self.embedding.compute_mask(*args, **kwargs)

  def call(self, x):
    length = tf.shape(x)[1]
    x = self.embedding(x)
    # This factor sets the relative scale of the embedding and positonal_encoding.
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x = x + self.pos_encoding[tf.newaxis, :length, :]
    return x

class SelfAttention(tf.keras.layers.Layer):
  def __init__(self, num_heads, key_dim, attention_axes):
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads,
    key_dim=key_dim, attention_axes=attention_axes)
    self.layernorm = tf.keras.layers.LayerNormalization()
    self.add = tf.keras.layers.Add()
  def call(self, x):
    attn_output = self.mha(
        query = x, key = x, value = x
    )
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x

class FeedForward(tf.keras.layers.Layer):
  def __init__(self, embed_size, dropout_rate=0.1):
    super().__init__()
    self.dense1 = tf.keras.layers.Dense(embed_size, activation='relu')
    self.dense2 = tf.keras.layers.Dense(embed_size, activation='relu')
    self.drop = tf.keras.layers.Dropout(dropout_rate)
    self.add = tf.keras.layers.Add()
    self.layer_norm = tf.keras.layers.LayerNormalization()

  def call(self, x):
    y = self.dense1(x)
    y = self.dense2(y)
    y = self.drop(y)
    x = self.add([x, y])
    x = self.layer_norm(x)
    return x

class BaseModel(tf.keras.Model):
  def __init__(self, vocab_size, maxlen, embed_size, num_heads, key_dim, attention_axes=(1,2)):
    super().__init__()
    self.positional_embedding = PositionalEmbedding(vocab_size, embed_size, maxlen)
    self.attention_block = SelfAttention(num_heads, key_dim, attention_axes)
  def call(self, x):
    x = self.positional_embedding(x)
    x = self.attention_block(x)
    return x

class QNetwork(tf.keras.Model):
  def __init__(self,
               vocab_size,
               embed_size,
               num_heads,
               key_dim,
               maxlen=29,
               base_model_wts = None,
               attention_axes=(1,2)):
    super().__init__()
    self.base_model = BaseModel(vocab_size, maxlen, embed_size, num_heads, key_dim, attention_axes = attention_axes)
    if base_model_wts != None:
      self.base_model.build(input_shape=(None,29))
      self.base_model.load_weights(base_model_wts)
    self.feed_forward = FeedForward(embed_size=embed_size)
    self.flat = tf.keras.layers.Flatten()
    self.out = tf.keras.layers.Dense(26, activation='linear')
  def call(self, state, guessed):
    #state: (batch_size, maxlen)
    #mask: (batch_size, 26)
    x = self.base_model(state)
    x = self.feed_forward(x)
    x = self.flat(x)
    x = self.out(x)
    x = tf.math.multiply(x, tf.cast(tf.math.logical_not(guessed), dtype=tf.float32)) #where guessed is true, values should be -inf
    return x
  

#model = build_q_network(vocab_size = 28, maxlen=29,embed_size = 128, num_heads = 16, key_dim = 2, hidden_units = 64)
# model = QNetwork(vocab_size = 28, maxlen=29,embed_size = 128, num_heads = 16, key_dim = 2, hidden_units = 64)
# word_src = "words_250000_train.txt"
# max_lives = 2
# num_env = 4
# env = Hangman(word_src, num_env=num_env, max_lives=max_lives, verbose = True)

# set_letters = set(string.ascii_lowercase)
# letters = list(set_letters)
# letters.sort()
# letter_dict = {l : i+1 for i, l in enumerate(letters)}
# letter_dict['_'] = 27
# letter_dict['.'] = 0

# def letter_to_num(c):
#   return letter_dict[c]

# def process_input(words):
#   seq = [list(map(letter_to_num, word)) for word in words]
#   return pad_sequences(seq, maxlen = 29, padding="post", value=0)

# state, guessed = env.reset()
# state = process_input(state)
# print(state.shape, guessed.shape)
# print(model(state, guessed))
