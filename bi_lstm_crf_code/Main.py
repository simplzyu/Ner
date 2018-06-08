#encoding=gbk
from sklearn.metrics import f1_score, accuracy_score
from Bi_Lstm_crf_model import Bi_Lstm_crf_model
from NER_Logger import NER_Logger
from Tools import next_batch
import tensorflow as tf
import numpy as np

class Main(object):
	def __init__(self, X, Y, X_val, Y_val, save_path, seq_length):
		self.model = Bi_Lstm_crf_model()
		self.save_path = save_path
		self.X = X
		self.Y = Y
		self.X_val = X_val
		self.Y_val = Y_val
		self.seq_length = seq_length
		self.logger = NER_Logger()

	def get_y_pre(self, sess, x, y):
		logits, transition_params = sess.run([self.model.logits, self.model.transition_params],
						  {self.model.input_x:x,self.model.input_y:y, self.model.dropout_keep_pro:1.0})
	  
		y_pre = []
		for logit in logits:
			sub_y, _ = tf.contrib.crf.viterbi_decode(logit, transition_params)
			y_pre += list(sub_y)
		return y_pre

	def evaluate(self, sess, x, y):
		
		y_pre, y_true = get_y_pre(sess, x, y), y.reshape([-1])
		for logit in logits:
			sub_y, _ = tf.contrib.crf.viterbi_decode(logit, transition_params)
			y_pre += list(sub_y)
		
		score = f1_score(y_true, y_pre, [0.,1.,2.,3.], average=None)
		acc = accuracy_score(y_true, y_pre)
		return  str(round(acc,3)) + ' | C_O:f1_score:%.3f, C_B:f1_score:%.3f, C_M:f1_score:%.3f, C_E:f1_score:%0.3f' % tuple(score)
	
	def evaluate2(self, y_true, y_pre):
		score = f1_score(y_true, y_pre, [0.,1.,2.,3.], average=None)
		acc = accuracy_score(y_true, y_pre)
		score = np.insert(score, 0, acc)
		return score

	#训练模型
	def train(self, num_epochs):
		saver = tf.train.Saver()
		init = tf.global_variables_initializer()
		with tf.Session() as sess:
			sess.run(init)
			for epoch in range(num_epochs):
				for xs,ys in next_batch(self.X, self.Y):
					sess.run(self.model.train_step, feed_dict = \
							 {self.model.input_x:xs, self.model.input_y:ys, self.model.dropout_keep_pro:0.8})

				self.model.lr *= self.model.lr_decay
				train_acc = self.evaluate(sess, self.X, self.Y)
				test_acc = self.evaluate(sess, self.X_val, self.Y_val)
				print('Iter ' + str(epoch) + ' Training Accuracy:' + train_acc)
				print('Iter ' + str(epoch) + ' Testing  Accuracy:' + test_acc)
			print(self.save_path)
			saver.save(sess=sess, save_path=self.save_path)
	
	def test(self, id2word, id2tag):
		saver = tf.train.Saver()
		init = tf.global_variables_initializer()
		with tf.Session() as sess:
			sess.run(init)
			self.logger.info('load model:' + self.save_path)
			saver.restore(sess, self.save_path)
			y_pre = self.get_y_pre(sess, self.X_val, self.Y_val)
			y_true = self.Y_val.reshape([-1])
			val_score = self.evaluate2(y_true, y_pre)
			val_info = 'Testing  Accuracy:%.3f | c0 f1_score:%.3f, c1 f1_score:%.3f, c2 f1_score:%.3f, c3 f1_score:%0.3f' % tuple(val_score)
			self.logger.info(val_info)	
			count = 0
			for i in range(len(self.X_val)):
				line, labels = self.X_val[i], self.Y_val[i]
				for j in range(len(line)):
					count += 1
					if count % self.seq_length == 0: print('')
					if line[j] == 0: continue
					print(id2word[line[j]], id2tag[labels[j]], id2tag[int(y_pre[count-1])])

			'''
			variable_names = [v.name for v in tf.trainable_variables()]
			values = sess.run(variable_names)
			for k,v in zip(variable_names, values):
				print("Variable: ", k)
				print("Shape: ", v.shape)
				print(v)	
			print('enter')
			'''
