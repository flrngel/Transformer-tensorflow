import tensorflow as tf
from flags import FLAGS

if FLAGS.dataset == 'dummy':
  file_base = './data/dummy/'
  if FLAGS.mode == 'train':
    file_a = file_base+'train.a.txt'
    file_b = file_base+'train.b.txt'

src = tf.data.TextLineDataset(file_a)
tgt = tf.data.TextLineDataset(file_b)

data = tf.data.Dataset.zip((src, tgt)).batch(FLAGS.batch_size)
data = data.make_initializable_iterator()
