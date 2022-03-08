import tensorflow as tf
import tensorflow.contrib.rnn as rnn

from data_utils import batch_review_normalize, batch_image_normalize
from layers import bidirectional_rnn, text_attention, visual_aspect_gate_unit
from model_utils import get_shape, load_glove

from data_preprocess import VOCAB_SIZE


class VistaNet:

  def __init__(self, hidden_dim, att_dim, emb_size, num_images, num_classes):
    self.hidden_dim = hidden_dim
    self.att_dim = att_dim
    self.emb_size = emb_size
    self.num_classes = num_classes
    self.num_images = num_images

    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    self.dropout_keep_prob = tf.placeholder(dtype=tf.float32, name='dropout_keep_prob')

    #[32,30,30]
    self.documents = tf.placeholder(shape=(None, None, None), dtype=tf.int32, name='reviews')
    #[30]
    self.document_lengths = tf.placeholder(shape=(None,), dtype=tf.int32, name='review_lengths')
    #[32,30]
    self.sentence_lengths = tf.placeholder(shape=(None, None), dtype=tf.int32, name='sentence_lengths')

    #30
    self.max_num_words = tf.placeholder(dtype=tf.int32, name='max_num_words')
    #30
    self.max_num_sents = tf.placeholder(dtype=tf.int32, name='max_num_sents')

    self.images = tf.placeholder(shape=(None, None, 4096), dtype=tf.float32, name='images')
    self.labels = tf.placeholder(shape=(None), dtype=tf.int32, name='labels')
    # self.global_mean_img = tf.Variable(initial_value=tf.random_normal(shape=[1, 1, 1, 100], stddev=0.1), trainable=True)

    with tf.variable_scope('VistaNet'):

      self._init_embedding()
      self._init_word_encoder()
      self._init_sent_encoder()
      self._init_classifier()

  def _init_embedding(self):
    with tf.variable_scope('embedding'):
      self.embedding_matrix = tf.get_variable(
        name='embedding_matrix',
        shape=[VOCAB_SIZE, self.emb_size],
        initializer=tf.constant_initializer(load_glove(VOCAB_SIZE, self.emb_size)),
        dtype=tf.float32
      )
      #[32,30,30,200]
      self.embedded_inputs = tf.nn.embedding_lookup(self.embedding_matrix, self.documents)

  def _init_word_encoder(self):
    with tf.variable_scope('word') as scope:
      #[960,30,200]
      self.word_rnn_inputs = tf.reshape(
        self.embedded_inputs,
        [-1, self.max_num_words, self.emb_size]
      )
      sentence_lengths = tf.reshape(self.sentence_lengths, [-1])

      # word encoder
      cell_fw = rnn.GRUCell(self.hidden_dim) #hidden_dim 50
      cell_bw = rnn.GRUCell(self.hidden_dim)

      init_state_fw = tf.tile(tf.get_variable('init_state_fw',
                                              shape=[1, self.hidden_dim],
                                              initializer=tf.constant_initializer(1.0)),
                              multiples=[get_shape(self.word_rnn_inputs)[0], 1])
      init_state_bw = tf.tile(tf.get_variable('init_state_bw',
                                              shape=[1, self.hidden_dim],
                                              initializer=tf.constant_initializer(1.0)),
                              multiples=[get_shape(self.word_rnn_inputs)[0], 1])

      #[960,30,100]
      self.word_rnn_outputs, _ = bidirectional_rnn(
        cell_fw=cell_fw,
        cell_bw=cell_bw,
        inputs=self.word_rnn_inputs,
        input_lengths=sentence_lengths,
        initial_state_fw=init_state_fw,
        initial_state_bw=init_state_bw,
        scope=scope
      )

      #word_outputs[960,100]
      self.word_outputs, self.word_att_weights = text_attention(inputs=self.word_rnn_outputs,
                                                                att_dim=self.att_dim,
                                                                sequence_lengths=sentence_lengths)

      self.word_outputs = tf.nn.dropout(self.word_outputs, keep_prob=self.dropout_keep_prob)

  def _init_sent_encoder(self):
    with tf.variable_scope('sentence') as scope:
      #[bs,ms,100]->[32,30,100]
      sentence_rnn_inputs = tf.reshape(self.word_outputs, [-1, self.max_num_sents, 2 * self.hidden_dim])

      # sentence encoders
      cell_fw_1 = rnn.GRUCell(self.hidden_dim, name='cell_fw_1')
      cell_bw_1 = rnn.GRUCell(self.hidden_dim, name='cell_bw_1')

      cell_fw_2 = rnn.GRUCell(self.hidden_dim, name='cell_fw_2')
      cell_bw_2 = rnn.GRUCell(self.hidden_dim, name='cell_bw_2')

      init_state_fw_1 = tf.tile(tf.get_variable('init_state_fw_1',
                                                shape=[1, self.hidden_dim],
                                                initializer=tf.constant_initializer(1.0)),
                                multiples=[get_shape(sentence_rnn_inputs)[0], 1])
      init_state_bw_1 = tf.tile(tf.get_variable('init_state_bw_1',
                                                shape=[1, self.hidden_dim],
                                                initializer=tf.constant_initializer(1.0)),
                                multiples=[get_shape(sentence_rnn_inputs)[0], 1])

      init_state_fw_2 = tf.tile(tf.get_variable('init_state_fw_2',
                                                shape=[1, self.hidden_dim],
                                                initializer=tf.constant_initializer(1.0)),
                                multiples=[get_shape(sentence_rnn_inputs)[0], 1])
      init_state_bw_2 = tf.tile(tf.get_variable('init_state_bw_2',
                                                shape=[1, self.hidden_dim],
                                                initializer=tf.constant_initializer(1.0)),
                                multiples=[get_shape(sentence_rnn_inputs)[0], 1])

      sentence_rnn_outputs_1, _ = bidirectional_rnn(
        cell_fw=cell_fw_1,
        cell_bw=cell_bw_1,
        inputs=sentence_rnn_inputs,
        input_lengths=self.document_lengths,
        initial_state_fw=init_state_fw_1,
        initial_state_bw=init_state_bw_1,
        scope=scope
      )

      sentence_rnn_outputs_2, _ = bidirectional_rnn(
        cell_fw=cell_fw_2,
        cell_bw=cell_bw_2,
        inputs=sentence_rnn_inputs,
        input_lengths=self.document_lengths,
        initial_state_fw=init_state_fw_2,
        initial_state_bw=init_state_bw_2,
        scope=scope
      )

      # self.sentence_outputs, self.sent_att_weights, self.img_att_weights = visual_aspect_attention(
      #     text_input=sentence_rnn_outputs_1,
      #     visual_input=self.images,
      #     att_dim=self.att_dim,
      #     sequence_lengths=self.document_lengths
      #   )

      #self.sentence_outputs = sentence_max_pooling(
        #text_input=sentence_rnn_outputs_1,
        #visual_input=self.images,
        #att_dim=self.att_dim,
        #sequence_lengths=self.document_lengths
      #)
      #
      # self.sentence_outputs = visual_aspect_gate_unit_v2(
      #     text_input_1=sentence_rnn_outputs_1,
      #     text_input_2=sentence_rnn_outputs_2,
      #     visual_input=self.images,
      #     att_dim=self.att_dim,
      #     sequence_lengths=self.document_lengths,
      #     global_mean_img=self.global_mean_img
      #   )

      self.sentence_outputs = visual_aspect_gate_unit(
          text_input_1=sentence_rnn_outputs_1,
          text_input_2=sentence_rnn_outputs_2,
          visual_input=self.images,
          att_dim=self.att_dim,
          sequence_lengths=self.document_lengths
      )

      # self.sentence_outputs = sent_attention(
      #   text_input_1=sentence_rnn_outputs_1,
      #   visual_input=self.images,
      #   att_dim=self.att_dim,
      #   sequence_lengths=self.document_lengths
      # )

      #[32,100]
      self.sentence_outputs = tf.nn.dropout(self.sentence_outputs, keep_prob=self.dropout_keep_prob)

  def _init_classifier(self):
    with tf.variable_scope('classifier'):
      self.logits = tf.layers.dense(
        inputs=self.sentence_outputs,
        units=self.num_classes,
        name='logits'
      )

  def get_feed_dict(self, reviews, images, labels, dropout_keep_prob=1.0):
    norm_docs, doc_sizes, sent_sizes, max_num_sents, max_num_words = batch_review_normalize(reviews)
    fd = {
      self.documents: norm_docs,
      self.document_lengths: doc_sizes,
      self.sentence_lengths: sent_sizes,
      self.max_num_sents: max_num_sents,
      self.max_num_words: max_num_words,
      self.images: batch_image_normalize(images, self.num_images),
      self.labels: labels,
      self.dropout_keep_prob: dropout_keep_prob
    }
    return fd
