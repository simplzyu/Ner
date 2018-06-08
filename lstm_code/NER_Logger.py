#encoding=gbk

import logging

class NER_Logger(object):
	def __init__(self):
		self.logger = logging.getLogger('NER')
		self.logger.setLevel('INFO')

		self.formatter = logging.Formatter('[%(asctime)s][%(levelname)s] ## %(message)s')
		self.ch = logging.StreamHandler()
		self.ch.setFormatter(self.formatter)

		self.logger.addHandler(self.ch)
	
	def info(self, message):
		self.logger.info(message)
