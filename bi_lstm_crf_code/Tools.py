#encoding=gbk

import numpy as np
import codecs
import random

class Tools(object):
	def __init__(self, seq_length, words_path):
		self.pad_word = '<PAD>'
		self.pad_tag = 'O'
		self.seq_length = seq_length
		self.words = []
		self.word2id = {}
		self.id2word = {}
		self.tag2id = {}
		self.id2tag = {}

		self.get_tag2id()
		self.load_words(words_path)
		self.get_word_dict()
		self.get_id2tag()
	
	def get_tag2id(self):
		#self.tag2id = {'O':[1.,.0,.0,.0], 'B':[.0,1.,.0,.0], 'M':[.0,.0,1.,.0], 'E':[.0,.0,.0,1.]}
		self.tag2id = {'O':0, 'B':1, 'M':2, 'E':3}
	def get_id2tag(self):
		self.id2tag = {0:'O', 1:'B', 2:'M', 3:'E'}

	def load_words(self, path):
		
		with codecs.open(path, 'r', encoding='gbk') as fin:
			for line in fin:
				word = line.strip()
				if len(word) == 0: continue
				self.words.append(word)
				
	def get_word_dict(self):
		for index, word in enumerate(self.words, 1):
			self.word2id[word] = index
			self.id2word[index] = word
		self.word2id[self.pad_word] = 0
		self.id2word[0] = self.pad_word
	
	def padding_sequence(self, sequence, seq_length):
		train_list, tag_list = [], []
		#print(sequence)
		#print(seq_length)
		for i in range(seq_length):
			if i >= len(sequence):
				train_list.append(self.word2id[self.pad_word])
				tag_list.append(self.tag2id[self.pad_tag])
			else:
				try:
					train_list.append(self.word2id[sequence[i][0]])
					tag_list.append(self.tag2id[sequence[i][1]])
				except Exception as e:
					print(str(e))
					print(sequence)
		return train_list, tag_list
	
	def process_file(self, path):
		X, Y, sequence = [], [], []
		with codecs.open(path, 'r', encoding='gbk') as fin:
			for line in fin:
				if line == '\n':
					sub_x, sub_y = self.padding_sequence(sequence, self.seq_length)
					X.append(sub_x)
					Y.append(sub_y)
					sequence = []
				else:
					items = line.strip().split(' ')
					if len(items) > 1:
						sequence.append(items)
			return np.array(X), np.array(Y)
	
def next_batch(X, Y, batch_size=64):
	length = len(X)
	num_batch = length // batch_size
	indexs = list(range(length))
	random.shuffle(indexs)
	x_shuffle = X[indexs]
	y_shuffle = Y[indexs]
	
	for i in range(num_batch):
		start = i * batch_size
		end = min((i+1) * batch_size, length)
		yield x_shuffle[start:end], y_shuffle[start:end]
