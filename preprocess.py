import re
import tensorflow as tf
import operator
from flags import FLAGS

word_dict_a = {'<EOS>': 0, '<SOS>': 1}
word_dict_b = {'<EOS>': 0, '<SOS>': 1}

def parser(data):
  TOKENIZER_RE = re.compile(r"[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+", re.UNICODE)
  return TOKENIZER_RE.findall(data.lower())

def transform(arr, word_dict):
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
    file_vocab_a = file_base+'vocab.a.txt'
    file_vocab_b = file_base+'vocab.b.txt'
elif FLAGS.dataset == 'IWSLT16':
  file_base = './data/IWSLT16/'
  if FLAGS.mode == 'train':
    file_a = file_base+'train.tags.de-en.en'
    file_b = file_base+'train.tags.de-en.de'
    file_a_out = file_base+'train.en.ids.txt'
    file_b_out = file_base+'train.de.ids.txt'
    file_vocab_a = file_base+'vocab.en.txt'
    file_vocab_b = file_base+'vocab.de.txt'

for fin, fout, word_dict in [(file_a, file_a_out, word_dict_a),
    (file_b, file_b_out, word_dict_b)]:
  with open(fout, 'w') as f_out:
    with open(fin) as f:
      for line in f:
        if len(line) > 0 and line[0] == '<':
          continue
        word_ids = map(lambda x: str(x), padding(transform(parser(line), word_dict)))
        f_out.write(' '.join(word_ids) + '\n')

for file_vocab, word_dict in [(file_vocab_a, word_dict_a),
    (file_vocab_b, word_dict_b)]:
  with open(file_vocab, 'w') as f_out:
    keys = list(map(lambda x: x[0],
        sorted(word_dict.items(), key=operator.itemgetter(1))))
    for key in keys:
      f_out.write(key + '\n')
