import tensorflow as tf
from flags import FLAGS
from model import Transformer
from reader import data, source_vocab, target_vocab

input_vocab_size = len(source_vocab)
output_vocab_size = len(target_vocab)

# Initialize data
initializer = tf.contrib.layers.xavier_initializer()
embedding_i = tf.get_variable('embedding_i', shape=[input_vocab_size,
  FLAGS.d_model], initializer=initializer)
embedding_o = tf.get_variable('embedding_o', shape=[output_vocab_size,
  FLAGS.d_model], initializer=initializer)

inputs_op, outputs_op = data.get_next()

embed_inputs_op = tf.nn.embedding_lookup(embedding_i, inputs_op)
embed_outputs_op = tf.nn.embedding_lookup(embedding_o, outputs_op)

# Load Transformer
if FLAGS.use_pretrained_vec == True:
  model = Transformer()
else:
  model = Transformer(inputs=embed_inputs_op, outputs=embed_outputs_op,
      sparse_outputs=outputs_op)

model.build_graph(output_vocab_size)

# Train
with tf.Session() as sess:
  sess.run([tf.global_variables_initializer(), data.initializer])
  train_writer = tf.summary.FileWriter('./summary/train', sess.graph)

  step = 0
  feed_dict = {}

  while True:
    try:
      if FLAGS.use_pretrained_vec == True:
        inputs, outputs, embed_inputs, embed_outputs = sess.run(
            [inputs_op, outputs_op, embed_inputs_op, embed_outputs_op])
        feed_dict = {model.inputs: embed_inputs,
          model.outputs: embed_outputs, model.sparse_outputs: outputs}
        _, summary, loss, predict = sess.run([model.optimize_op,
          model.summary_op, model.loss, model.predict],
          feed_dict=feed_dict)
      else:
        _, summary, loss, predict, inputs, outputs = sess.run([model.optimize_op,
          model.summary_op, model.loss, model.predict, inputs_op, outputs_op],
          feed_dict=feed_dict)

      if step % 77 == 0:
        train_writer.add_summary(summary, step)

        predict = predict.tolist()
        original = []
        result = []

        for p_i in predict[0]:
          result.append(target_vocab[p_i])
        for p_i in outputs[0]:
          original.append(target_vocab[p_i])

        if '<EOS>' in result:
          result = result[:result.index('<EOS>')]
        if '<EOS>' in original:
          original = original[:original.index('<EOS>')]

        original = ' '.join(original)
        result = ' '.join(result)


        print('step:'+str(step)+', loss: ' + str(loss))
        print(original)
        print(result)
        print('---')

      step += 1
    except tf.errors.OutOfRangeError:
      print('train done')
      break
