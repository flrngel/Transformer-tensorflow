import tensorflow as tf
print(tf.__version__)
import numpy as np
from flags import FLAGS

'''
Transformer modules
'''

def add_and_norm(x, sub_x):
  with tf.variable_scope('add_and_norm'):
    sub_x = tf.nn.dropout(sub_x, FLAGS.dropout_keep)
    return tf.contrib.layers.layer_norm(x + sub_x)

def feed_forward(x, d_ff=2048):
  output_dim = x.get_shape()[-1]
  with tf.variable_scope('feed_forward'):
    x = tf.layers.dense(x, d_ff, activation=tf.nn.relu)
    x = tf.layers.dense(x, output_dim)
    return x

def multihead_attention_block(vk_input, q_input, 
    batch_size, pad_length, d_model, d_k, d_v, masked=False):

  with tf.variable_scope('multihead_attention'):
    K = tf.layers.dense(vk_input, d_k, name='K')
    V = tf.layers.dense(vk_input, d_v, name='V')
    Q = tf.layers.dense(q_input, d_k, name='Q')

    '''
    Scaled Dot-Product Attention
    '''
    # Mask (pad_length x pad_length)
    mask = tf.ones([pad_length, pad_length])
    if masked == True:
      #mask = tf.linalg.LinearOperatorLowerTriangular(mask, f.float32).to_dense()
      mask = tf.contrib.linalg.LinearOperatorTriL(mask, tf.float32).to_dense()
    mask = tf.reshape(tf.tile(mask, [batch_size, 1]),
        [batch_size, pad_length, pad_length])

    # Attention(Q,K,V)
    attn = tf.nn.softmax(
        mask * (Q @ tf.transpose(K, [0, 2, 1])) / tf.sqrt(tf.to_float(d_k))) @ V

    return attn

def multihead_attention(vk_input, q_input, masked=False):
  outputs = []

  pad_length = FLAGS.pad_length
  batch_size = tf.shape(vk_input)[0]
  d_model = FLAGS.d_model
  d_k = FLAGS.d_k
  d_v = FLAGS.d_v
  h = FLAGS.h

  for i in range(h):
    outputs.append(
        multihead_attention_block(vk_input, q_input,
          batch_size, pad_length, d_model, d_k, d_v, masked=masked))
  outputs = tf.concat(outputs, axis=2)
  outputs = tf.layers.dense(outputs, d_model)
  return outputs

'''
Transformer Encoder block
'''
def encoder_block(inputs):
  # load hyper parameters

  with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
    flow = multihead_attention(inputs, inputs)
    flow = add_and_norm(inputs, flow)
    flow = add_and_norm(flow, feed_forward(flow))
    return flow

'''
Transformer Decoder block
'''
def decoder_block(outputs, encoder_outputs):
  # load hyper parameters

  with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
    flow = multihead_attention(outputs, outputs, masked=True)
    flow = add_and_norm(outputs, flow)
    flow = add_and_norm(flow, multihead_attention(encoder_outputs, flow))
    flow = add_and_norm(flow, feed_forward(flow))
    return flow

'''
Positional Encoding
'''
def positional_encoding(x):
  pad_length = FLAGS.pad_length
  d_model = FLAGS.d_model

  def sincos(x, i):
    if i%2 == 0:
      return np.sin(x)
    return np.cos(x)

  with tf.variable_scope('positional_encoding'):
    pe = tf.convert_to_tensor([sincos(pos/(10000**(2*i/d_model)), i)
      for pos in range(1, pad_length+1) for i in range(1, d_model+1)])
    pe = tf.reshape(pe, [-1, pad_length, d_model])
    return tf.add(x, pe)

'''
Transformer class
'''

class Transformer(object):
  def __init__(self):
    pad_length = FLAGS.pad_length
    d_model = FLAGS.d_model

    self.inputs = tf.placeholder(tf.float32, shape=[None, pad_length, d_model])
    self.outputs = tf.placeholder(tf.float32, shape=[None, pad_length, d_model])
    self.sparse_outputs = tf.placeholder(tf.int32, shape=[None, pad_length])

  def build_graph(self):
    N = FLAGS.stack_num
    learn_rate = FLAGS.learn_rate
    dic_size = FLAGS.dic_size

    with tf.variable_scope('transformer'):
      inputs = positional_encoding(self.inputs)
      outputs = positional_encoding(self.outputs)

      for i in range(N):
        inputs = encoder_block(inputs)

      for i in range(N):
        outputs = decoder_block(outputs, inputs)

      with tf.variable_scope('projection'):
        self.logits = tf.layers.dense(outputs, dic_size)

        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
              labels=self.sparse_outputs,
              logits=self.logits))

      optimizer = tf.train.AdamOptimizer(learn_rate)
      self.optimize_op = optimizer.minimize(self.loss)
