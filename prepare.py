import tensorflow as tf
import operator
from polyglot.text import Text
from flags import FLAGS

word_dict = {'<EOS>': 0, '<SOS>': 1}

def parser(data):
  return Text(line).words

def transform(arr):
  result = []
  for i in range(len(arr)):
    if arr[i] not in word_dict:
      word_dict[arr[i]] = len(word_dict)
    result.append(word_dict[arr[i]])
  return result

def padding(arr):
  pad_length = FLAGS.pad_length
  result = tf.keras.preprocessing.sequence.pad_sequences([arr], pad_length, padding='post')
  return result[0]

if FLAGS.dataset == 'dummy':
  file_base = './data/dummy/'
  if FLAGS.mode == 'train':
    file_a = file_base+'train.a.txt'
    file_b = file_base+'train.b.txt'
    file_a_out = file_base+'train.a.ids.txt'
    file_b_out = file_base+'train.b.ids.txt'
    file_vocab = file_base+'vocab.txt'

for fin, fout in [(file_a, file_a_out), (file_b, file_b_out)]:
  with open(fout, 'w') as f_out:
    with open(fin) as f:
      line = f.readline()
      word_ids = map(lambda x: str(x), padding(transform(parser(line))))
      f_out.write(' '.join(word_ids) + '\n')

with open(file_vocab, 'w') as f_out:
  keys = list(map(lambda x: x[0],
      sorted(word_dict.items(), key=operator.itemgetter(1))))
  for key in keys:
    f_out.write(key + '\n')
