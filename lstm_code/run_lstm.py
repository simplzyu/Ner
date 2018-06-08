#encoding=gbk
from Main import Main
from Tools import Tools
import time
import sys

def run(is_train, num_epochs):
	#词表位置
	words_path = '../data/words.txt'
	
	#模型保存位置
	now_time = time.strftime("%Y%m%d_%H%M%S")		
	save_path = '../model/lstm_model/' + now_time + 'model.ckpt'
	#序列长度
	seq_length = 37
	
	if is_train == 'test':
		save_path = '../model/lstm_model/' + num_epochs

	tools = Tools(seq_length, words_path)
	X,Y = tools.process_file('../data/bendibao.train')
	X_val, Y_val = tools.process_file('../data/bendibao.test')
	
	main = Main(X, Y, X_val, Y_val, save_path, seq_length)
	if is_train == 'train':
		main.train(num_epochs)
	elif is_train == 'test':
		main.test(tools.id2word, tools.id2tag)
	else:
		print('parameters error')

def main():
	is_train = sys.argv[1]
	num_epochs = 0
	if len(sys.argv) > 1:
		if is_train == 'train':
			num_epochs = int(sys.argv[2])
		elif is_train == 'test':
			num_epochs = sys.argv[2]
		else:
			print('parameters error')
			eixt(0)
	run(is_train, num_epochs)

if __name__ == '__main__':
	main()
