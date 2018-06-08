#encoding=gbk
import tensorflow as tf

class Bi_Lstm_crf_model(object):
	def __init__(self):
		self.batch_size = 64
		self.seq_length = 37
		self.num_classes = 4
		self.lr = 0.02
		self.lr_decay = 0.9
		self.dropout_keep_pro = 0.8
		self.num_epoch = 20
		self.hidden_dim = 128
		self.num_layer = 2
		self.embedding_dim = 64
		self.vocab_size = 1745
		
		self.run()
		
	def run(self):
		#inpux
		tf.reset_default_graph()
		self.input_x = tf.placeholder(dtype=tf.int32, shape=[None, self.seq_length], name='input_x')
		self.input_y = tf.placeholder(dtype=tf.int32, shape=[None, self.seq_length], name='input_y')
		self.sequence_lengths = tf.constant([self.seq_length] * self.batch_size, dtype=tf.int32)
		self.dropout_keep_pro = tf.placeholder(dtype=tf.float32, name='keep_pro')
		
		#input_y 变形
		#self.labels = tf.reshape(self.input_y, [-1])
		
		#lstm
		def lstm_cell():
			return tf.contrib.rnn.LSTMCell(self.hidden_dim)
		
		def dropout():
			cell = lstm_cell()
			return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.dropout_keep_pro)
		
		def get_weight(shape):
			return tf.Variable(tf.random_normal(shape=shape, stddev=0.1))
		
		#embedding
		with tf.name_scope('embedding'):
			embedding = tf.get_variable('embedding', shape=[self.vocab_size, self.embedding_dim], dtype=tf.float32)
			embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)
	
		with tf.name_scope('bi_lstm'):
			#cells = [dropout() for i in range(self.num_layer)]
			#rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
			#fw_cell, bw_cell = lstm_cell(), lstm_cell
			# lstm模型　正方向传播的RNN
			lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim, forget_bias=1.0)
			# 反方向传播的RNN
			lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim, forget_bias=1.0)
			#print(fw_cell)
			(fw_outputs, bw_outputs),_ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, embedding_inputs, dtype=tf.float32)
			outputs = tf.concat([fw_outputs, bw_outputs], axis= 2)
			
			#outputs,_ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=embedding_inputs, dtype=tf.float32)
			'''
			print(embedding_inputs.shape)
			print(outputs.shape)
			print(outputs[:,-1,:].shape)
			'''
			#改变形状
			outputs = tf.reshape(outputs, [-1, 2 * self.hidden_dim])
		
		with tf.name_scope('full_layer'):
			weight1 = get_weight(shape=[2 * self.hidden_dim, self.hidden_dim])
			biases = tf.Variable(tf.constant(0.1,dtype=tf.float32, shape = [self.hidden_dim]))
			fc1 = tf.matmul(outputs, weight1) + biases
			fc1 = tf.nn.dropout(fc1, keep_prob=self.dropout_keep_pro)
			fc1 = tf.nn.relu(fc1)
			
			weigth2 = get_weight(shape=[self.hidden_dim, self.num_classes])
			biases2 = tf.Variable(tf.constant(0.1,dtype=tf.float32, shape = [self.num_classes]))
			self.logits = tf.matmul(fc1, weigth2) + biases2
			self.logits = tf.reshape(self.logits, [-1,self.seq_length, self.num_classes])
		
		with tf.name_scope('crf_layer'):
			#print(self.logits.shape, self.input_y.shape)
			# crf 得到维特比矩阵 
			log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(\
							self.logits, self.input_y, self.sequence_lengths)
			
		with tf.name_scope('train_step'):
			#print('labels',self.labels.shape)
			#print('logits',self.logits.shape)
			#cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits)
			#tf.nn.sparse_softmax_cross_entropy_with_logits(
			loss = tf.reduce_mean(-log_likelihood)
			self.train_step = tf.train.GradientDescentOptimizer(self.lr).minimize(loss)
