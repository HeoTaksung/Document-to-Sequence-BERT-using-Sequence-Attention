import tensorflow as tf
from tensorflow.python.keras import backend as K
from transformers import *


class Sequence_Attention(tf.keras.layers.Layer):
    def __init__(self, class_num):
        super(Sequence_Attention, self).__init__(class_num)
        self.class_num = class_num
        self.Ws = None

    def build(self, input_shape):
        embedding_length = int(input_shape[2])

        self.Ws = self.add_weight(shape=(self.class_num, embedding_length),
                                  initializer=tf.keras.initializers.get('glorot_uniform'), trainable=True)

        super(Sequence_Attention, self).build(input_shape)

    def call(self, inputs):

        sentence_trans = tf.transpose(inputs, [0, 2, 1])

        at = tf.matmul(self.Ws, sentence_trans)

        at = tf.math.tanh(at)

        at = K.exp(at - K.max(at, axis=-1, keepdims=True))
        at = at / K.sum(at, axis=-1, keepdims=True)

        v = K.batch_dot(at, inputs)

        return v


class D2SBERT_Model(tf.keras.Model):
    def __init__(self, model_name, dir_path, num_class):
        super(D2SBERT_Model, self).__init__()

        self.num_class = num_class

        self.bert = TFBertModel.from_pretrained(model_name, cache_dir=dir_path, output_attentions=True, from_pt=True)

        self.att = Sequence_Attention(self.num_class)

        self.dropout = tf.keras.layers.Dropout(self.bert.config.hidden_dropout_prob)

        self.classifier = [tf.keras.layers.Dense(1, activation='sigmoid',
                                                 kernel_initializer=tf.keras.initializers.TruncatedNormal(
                                                     self.bert.config.initializer_range))
                           for _ in range(self.num_class)]

    def call(self, inputs, attention_mask=None, token_type_ids=None, training=False):

        ids = tf.keras.layers.Reshape((25, 512))(inputs[0])
        mask = tf.keras.layers.Reshape((25, 512))(inputs[1])
        token = tf.keras.layers.Reshape((25, 512))(inputs[2])

        ids = tf.transpose(ids, [1, 0, 2])
        mask = tf.transpose(mask, [1, 0, 2])
        token = tf.transpose(token, [1, 0, 2])

        CLS_tokens = []

        for i in range(ids.shape[0]):
            CLS_tokens.append(self.bert(ids[i], attention_mask=mask[i], token_type_ids=token[i])[1])

        CLS_tokens = tf.stack(CLS_tokens, axis=1)

        CLS_tokens = self.att(CLS_tokens)

        labels = []

        for i in range(self.num_class):
            labels.append(self.classifier[i](self.dropout(CLS_tokens[::, i])))

        labels = tf.stack(labels, axis=1)

        labels = tf.squeeze(labels, [2])

        return labels
