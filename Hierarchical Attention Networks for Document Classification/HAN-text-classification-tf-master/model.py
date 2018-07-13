#coding=utf8

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import layers

def length(sequences):
    used = tf.sign(tf.reduce_max(tf.abs(sequences), reduction_indices=2))   #1920 * 30      #为什么求的是嵌入的词向量中的最大值，seq_len不应该是所有句子的长度吗？
    seq_len = tf.reduce_sum(used, reduction_indices=1)                      #1 * 1920
    print (seq_len.shape)
    print (used.shape)
    return tf.cast(seq_len, tf.int32)

class HAN():

    def __init__(self, vocab_size, num_classes, embedding_size=200, hidden_size=50):

        self.vocab_size = vocab_size              #46960
        self.num_classes = num_classes            #5
        self.embedding_size = embedding_size      #200
        self.hidden_size = hidden_size            #50

        with tf.name_scope('placeholder'):
            self.max_sentence_num = tf.placeholder(tf.int32, name='max_sentence_num')
            self.max_sentence_length = tf.placeholder(tf.int32, name='max_sentence_length')
            self.batch_size = tf.placeholder(tf.int32, name='batch_size')
            #x的shape为[batch_size, 句子数， 句子长度(单词个数)]，但是每个样本的数据都不一样，，所以这里指定为空
            #y的shape为[batch_size, num_classes]
            self.input_x = tf.placeholder(tf.int32, [None, None, None], name='input_x')     #64 * 30 * 30
            self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')   #64 * 5

        #构建模型
        word_embedded = self.word2vec()
        sent_vec = self.sent2vec(word_embedded)
        doc_vec = self.doc2vec(sent_vec)
        out = self.classifer(doc_vec)

        self.out = out


    def word2vec(self):
        with tf.name_scope("embedding"):
            embedding_mat = tf.Variable(tf.truncated_normal((self.vocab_size, self.embedding_size)))   #46960 * 200
            #shape为[batch_size, sent_in_doc, word_in_sent, embedding_size]
            word_embedded = tf.nn.embedding_lookup(embedding_mat, self.input_x)                        #64 * 30 * 30 * 200
        return word_embedded

    def sent2vec(self, word_embedded):
        with tf.name_scope("sent2vec"):
            #GRU的输入tensor是[batch_size, max_time, ...].在构造句子向量时max_time应该是每个句子的长度，所以这里将
            #batch_size * sent_in_doc当做是batch_size.这样一来，每个GRU的cell处理的都是一个单词的词向量
            #并最终将一句话中的所有单词的词向量融合（Attention）在一起形成句子向量

            #shape为[batch_size*sent_in_doc, word_in_sent, embedding_size]
            word_embedded = tf.reshape(word_embedded, [-1, self.max_sentence_length, self.embedding_size])  #1920 * 30 * 200  #
            #shape为[batch_size*sent_in_doce, word_in_sent, hidden_size*2]
            word_encoded = self.BidirectionalGRUEncoder(word_embedded, name='word_encoder')  #1920 * 30 * 100
            #shape为[batch_size*sent_in_doc, hidden_size*2]
            sent_vec = self.AttentionLayer(word_encoded, name='word_attention')              #1920 * 100
            return sent_vec

    def doc2vec(self, sent_vec):
        with tf.name_scope("doc2vec"):
            sent_vec = tf.reshape(sent_vec, [-1, self.max_sentence_num, self.hidden_size*2])  #64 * 30 * 100
            #shape为[batch_size, sent_in_doc, hidden_size*2]
            doc_encoded = self.BidirectionalGRUEncoder(sent_vec, name='sent_encoder')      #64 * 30 * 100
            #shape为[batch_szie, hidden_szie*2]
            doc_vec = self.AttentionLayer(doc_encoded, name='sent_attention')              #64 * 100
            return doc_vec

    def classifer(self, doc_vec):
        with tf.name_scope('doc_classification'):
            out = layers.fully_connected(inputs=doc_vec, num_outputs=self.num_classes, activation_fn=None)  #64 * 5
            return out

    def BidirectionalGRUEncoder(self, inputs, name):
        #输入inputs的shape是[batch_size, max_time, voc_size]   #1920 * 30 * 200    #64 * 30 * 100
        with tf.variable_scope(name):
            GRU_cell_fw = rnn.GRUCell(self.hidden_size)    #输出分别代表前向GRU和后向GRU的cell
            GRU_cell_bw = rnn.GRUCell(self.hidden_size)
            #fw_outputs和bw_outputs的size都是[batch_size, max_time, hidden_size]
            ((fw_outputs, bw_outputs), (_, _)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=GRU_cell_fw,
                                                                                 cell_bw=GRU_cell_bw,
                                                                                 inputs=inputs,
                                                                                 sequence_length=length(inputs),
                                                                                 dtype=tf.float32)
            #outputs的size是[batch_size, max_time, hidden_size*2]
            outputs = tf.concat((fw_outputs, bw_outputs), 2)   #1920 * 30 * 100    #64 * 30 * 100
            return outputs
    def variable_summaries(self,var):
        tf.summary.histogram('histogram', var)
    def AttentionLayer(self, inputs, name):
        #inputs是GRU的输出，size是[batch_size, max_time, encoder_size(hidden_size * 2)]  #1920 * 30 * 100   #64 * 30 * 100
        with tf.variable_scope(name):
            # u_context是上下文的重要性向量，用于区分不同单词/句子对于句子/文档的重要程度,
            # 因为使用双向GRU，所以其长度为2×hidden_szie
            u_context = tf.Variable(tf.truncated_normal([self.hidden_size * 2]), name='u_context')
            self.variable_summaries(u_context)
            #使用一个全连接层编码GRU的输出的到其隐层表示,输出u的size是[batch_size, max_time, hidden_size *  2]
            h = layers.fully_connected(inputs, self.hidden_size * 2, activation_fn=tf.nn.tanh)  #1920 * 30 * 100    #64 * 30 * 100
            #shape为[batch_size, max_time, 1]
            alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(h, u_context), axis=2, keep_dims=True), dim=1)  #1920 * 30 * 1   #64 * 30 * 1
            #reduce_sum之前shape为[batch_szie, max_time, hidden_szie*2]，之后shape为[batch_size, hidden_size*2]
            atten_output = tf.reduce_sum(tf.multiply(inputs, alpha), axis=1)     #1920 * 100   #64 * 100
            return atten_output


