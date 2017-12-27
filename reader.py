import tensorflow as tf
from flags import FLAGS

if FLAGS.dataset == 'dummy':
  file_base = './data/dummy/'
  if FLAGS.mode == 'train':
    file_a = file_base+'train.a.ids.txt'
    file_b = file_base+'train.b.ids.txt'
    vocab_source_filename = file_base+'vocab.a.txt'
    vocab_target_filename = file_base+'vocab.b.txt'
elif FLAGS.dataset == 'IWSLT16':
  file_base = './data/IWSLT16/'
  if FLAGS.mode == 'train':
    file_a = file_base+'train.de.ids.txt'
    file_b = file_base+'train.en.ids.txt'
    vocab_source_filename = file_base+'vocab.de.txt'
    vocab_target_filename = file_base+'vocab.en.txt'

src = tf.data.TextLineDataset(file_a).map(lambda x: tf.string_split([x]).values).map(lambda x: tf.string_to_number(x, tf.int32))
tgt = tf.data.TextLineDataset(file_b).map(lambda x: tf.string_split([x]).values).map(lambda x: tf.string_to_number(x, tf.int32))

data = tf.data.Dataset.zip((src, tgt)).repeat(FLAGS.num_epochs).batch(FLAGS.batch_size)
data = data.make_initializable_iterator()

source_vocab = []
with open(vocab_source_filename) as f:
  for line in f:
    source_vocab.append(line.strip())

target_vocab = []
with open(vocab_target_filename) as f:
  for line in f:
    target_vocab.append(line.strip())
