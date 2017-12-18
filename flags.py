import tensorflow as tf

tf.app.flags.DEFINE_string('mode', 'train', 'mode to train/test')
tf.app.flags.DEFINE_integer('num_epochs', 5, 'number of epochs to train')
tf.app.flags.DEFINE_integer('batch_size', 32, 'batch size to train in one step')
tf.app.flags.DEFINE_string('dataset', 'dummy', 'dataset. supports iwslt16.')

FLAGS = tf.app.flags.FLAGS
