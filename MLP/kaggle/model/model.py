import numpy as np 
import tensorflow as tf 
import sklearn
import pickle
import math
import config as cfg 
import argparse

class MLP(object):
	def __init__(self):
		self.layer = cfg.LAYER 
		self.learning_rate = cfg.LEARNING_RATE 
		self.input = tf.placeholder(shape = [None, 40], dtype = tf.float32, name = 'inputs')
		self.label = tf.placeholder(shape = [None, 138], dtype = tf.float32, name = 'labels')

	def perceptron_layer(self, idx_layer, input, activation = 'sigmoid'):
		with tf.variable_scope("pl" + str(idx_layer)) as scope:
			# weights = tf.get_variable(shape = [input.shape[1],self.layer[idx_layer-1]], dtype = tf.float32, name = 'weights')
			weights = tf.get_variable(shape = [input.shape[1],self.layer[idx_layer-1]], dtype = tf.float32, initializer = tf.truncated_normal_initializer(), name = 'weights')
			bias    = tf.get_variable(shape = [self.layer[idx_layer - 1]],dtype = tf.float32,  name = 'bias')
		if(activation == 'softmax'):
			return tf.add(tf.matmul(input, weights), bias)
		return tf.sigmoid(tf.add(tf.matmul(input, weights), bias))


	def build_model(self):
		self.net = self.input
		for i in range(1,len(self.layer)):
			self.net = self.perceptron_layer(i, self.net)
		self.net = self.perceptron_layer(len(self.layer), self.net, activation = 'softmax')
		self.sess = tf.Session()
		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.net, labels = tf.cast(self.label, dtype = tf.float32)))
		# self.optimizer = tf.train.GradientDescentOptimizer(learning_rate = self.learning_rate).minimize(self.loss)
		self.optimizer = tf.train.AdagradOptimizer(learning_rate = self.learning_rate).minimize(self.loss)
		self.init = tf.global_variables_initializer()
		self.saver = tf.train.Saver()
		self.writer = tf.summary.FileWriter('../my_graph', self.sess.graph)
		self.writer.close()
		print("model completed.")

	def load_data(self, mode):
		if(mode == 'train'):
			data = np.load('../data/dev.npy', encoding = 'latin1')
			label = np.load('../data/dev_labels.npy', encoding = 'latin1')
		elif(mode == 'test'):
			data = np.load('../data/dev.npy', encoding = 'latin1')
			label = np.load('../data/dev_labels.npy', encoding = 'latin1')
		else:
			raise Exception('Wrong mode.')
		label_one_hot = []
		for i_label in label:
			one_label = np.zeros([i_label.shape[0],138])
			for i in range(i_label.shape[0]):
				one_label[i, i_label[i]] = 1
			label_one_hot.append(one_label)
		return data, label_one_hot
		

	def train(self):
		data, label = self.load_data('train')
		self.sess.run(self.init)
		for epoch in range(5):
			avg_loss = 0
			for idx_batch in range(len(data)):
				data_batch = data[idx_batch]
				label_batch = label[idx_batch]
				_, cost = self.sess.run((self.optimizer, self.loss), feed_dict = {self.input:data_batch, self.label:label_batch})
				avg_loss += cost
			print("epoch %04d : loss = %.9f"%(epoch, avg_loss))
		self.saver.save(self.sess, '../weights/mlp.ckpt')
		print('training completed.')

	def test(self):
		data, label = self.load_data('test')
		self.saver.restore(self.sess, '../weights/mlp.ckpt')
		correct_prediction = tf.equal(tf.argmax(self.net, axis = 1), tf.argmax(self.label, axis = 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype = tf.float32))
		avg_acc = 0
		for idx_batch in range(len(data)):
			data_batch = data[idx_batch]
			label_batch = label[idx_batch]
			acc = self.sess.run(accuracy, feed_dict = {self.input:data_batch, self.label:label_batch})
			avg_acc += acc
		avg_acc /= len(data)
		print("Total average accuracy = " + str(avg_acc))

	def retrain(self):
		data, label = self.load_data('train')
		self.saver.restore(self.sess, '../weights/mlp.ckpt')
		for epoch in range(1000):
			avg_loss = 0
			for idx_batch in range(len(data)):
				data_batch = data[idx_batch]
				label_batch = label[idx_batch]
				_, cost = self.sess.run((self.optimizer, self.loss), feed_dict = {self.input:data_batch, self.label:label_batch})
				avg_loss += cost
			print("epoch %04d : loss = %.9f"%(epoch, avg_loss))
		self.saver.save(self.sess, '../weights/mlp.ckpt')
		print('training completed.')


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', type = str)
	args = parser.parse_args()
	mlp = MLP()
	mlp.build_model()
	if(args.mode == 'train'):
		mlp.train()
	elif(args.mode == 'test'):
		mlp.test()
	elif(args.mode == 'retrain'):
		mlp.retrain()
	else:
		raise Exception('Wrong mode.')

if __name__ == "__main__":
	main()

