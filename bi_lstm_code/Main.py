#encoding=gbk
from sklearn.metrics import f1_score
from Bi_Lstm_model import Bi_Lstm_model
from NER_Logger import NER_Logger
from Tools import next_batch
import tensorflow as tf
import numpy as np

class Main(object):
	def __init__(self, X, Y, X_val, Y_val, save_path, seq_length):
		self.model = Bi_Lstm_model()
		self.save_path = save_path
		self.X = X
		self.Y = Y
		self.X_val = X_val
		self.Y_val = Y_val
		self.seq_length = seq_length
		self.logger = NER_Logger()

	def evaluate(self, sess, x, y):
		y_true = sess.run(self.model.y_true, feed_dict=\
						  {self.model.input_x:x, self.model.input_y:y, self.model.dropout_keep_pro:1.0})
		y_pre = sess.run(self.model.y_pre, feed_dict=\
						 {self.model.input_x:x, self.model.input_y:y, self.model.dropout_keep_pro:1.0})
		score = f1_score(y_true, y_pre, [0.,1.,2.,3.], average=None)
		return 'C_O:f1_score:%.3f, C_B:f1_score:%.3f, C_M:f1_score:%.3f, C_E:f1_score:%0.3f' % tuple(score)

	#训练模型
	def train(self, num_epochs):
		saver = tf.train.Saver()
		init = tf.global_variables_initializer()
		with tf.Session() as sess:
			sess.run(init)
			print(self.X.shape)
			print(self.Y.shape)
			for epoch in range(num_epochs):
				for xs,ys in next_batch(self.X, self.Y):
					#print(xs.shape)
					#print(ys)
					sess.run(self.model.train_step, feed_dict = \
							 {self.model.input_x:xs, self.model.input_y:ys, self.model.dropout_keep_pro:0.8})
					#labels = sess.run(model.labels,feed_dict = {model.input_x:xs, model.input_y:ys, model.dropout_keep_pro:0.8})
					#logits = sess.run(model.logits,feed_dict = {model.input_x:xs, model.input_y:ys, model.dropout_keep_pro:0.8})
					#acc = sess.run(model.accuracy, feed_dict = {model.input_x:xs, model.input_y:ys, model.dropout_keep_pro:0.8})

					#print('Iter: ' + str(epoch) + ', Training Accuracy:' + str(acc))
				self.model.lr *= self.model.lr_decay
				train_acc = sess.run(self.model.accuracy, feed_dict = \
					{self.model.input_x:self.X, self.model.input_y:self.Y, self.model.dropout_keep_pro:1.0})
				test_acc = sess.run(self.model.accuracy, feed_dict = \
					{self.model.input_x:self.X_val, self.model.input_y:self.Y_val, self.model.dropout_keep_pro:1.0})
				f1_score_str = self.evaluate(sess, self.X_val, self.Y_val)
				print('Iter: ' + str(epoch) + ', Training Accuracy:' + str(train_acc) + ', Testing Accuracy:'\
					  + str(test_acc) + ' | ' + f1_score_str)
			print(self.save_path)
			saver.save(sess=sess, save_path=self.save_path)
	
	def test(self, id2word, id2tag):
		saver = tf.train.Saver()
		init = tf.global_variables_initializer()
		with tf.Session() as sess:
			sess.run(init)
			self.logger.info('load model:' + self.save_path)
			saver.restore(sess, self.save_path)
			y_pre = sess.run(self.model.y_pre, feed_dict=\
							{self.model.input_x:self.X_val, self.model.dropout_keep_pro:1.0})
			y_true = sess.run(self.model.y_true, feed_dict=\
							{self.model.input_y:self.Y_val, self.model.dropout_keep_pro:1.0})
			acc = sess.run(self.model.accuracy, feed_dict = \
					{self.model.input_x:self.X_val, self.model.input_y:self.Y_val, self.model.dropout_keep_pro:1.0})

			score = list(f1_score(y_true, y_pre, [0.,1.,2.,3.], average=None))
			score.insert(0, acc)
			self.logger.info('Accuracy:%.3f | C_O:f1_score:%.3f, C_B:f1_score:%.3f, C_M:f1_score:%.3f, C_E:f1_score:%0.3f' % tuple(score))
			
			count = 0
			for i in range(len(self.X_val)):
				line, labels = self.X_val[i], self.Y_val[i]
				for j in range(len(line)):
					count += 1
					if count % self.seq_length == 0: print('')
					if line[j] == 0: continue
					print(id2word[line[j]], id2tag[np.argmax(labels[j], axis=0)], id2tag[int(y_pre[count-1])])

			'''
			variable_names = [v.name for v in tf.trainable_variables()]
			values = sess.run(variable_names)
			for k,v in zip(variable_names, values):
				print("Variable: ", k)
				print("Shape: ", v.shape)
				print(v)	
			print('enter')
			'''
