import tensorflow as tf

# app parameter
tf.app.flags.DEFINE_string('mode', 'train', 'mode to train/test')
tf.app.flags.DEFINE_string('dataset', 'dummy', 'dataset')

# model parameter
tf.app.flags.DEFINE_integer('stack_num', 6, 'stack num')
tf.app.flags.DEFINE_integer('d_model', 512, 'model dimension')
tf.app.flags.DEFINE_integer('d_k', 64, 'key dim')
tf.app.flags.DEFINE_integer('d_v', 64, 'value dim')
tf.app.flags.DEFINE_integer('h', 8, 'stack of multihead attention')

# train parmeter
tf.app.flags.DEFINE_integer('num_epochs', 5, 'num epochs')
tf.app.flags.DEFINE_integer('batch_size', 32, 'batch size')

tf.app.flags.DEFINE_float('dropout_keep', 0.9, 'dropout keep rate')
tf.app.flags.DEFINE_integer('pad_length', 60, 'pad length')
tf.app.flags.DEFINE_float('learn_rate', 1e-4, 'learn rate')
tf.app.flags.DEFINE_boolean('use_pretrained_vec', False, 'flag for pretrained vector')

FLAGS = tf.app.flags.FLAGS
