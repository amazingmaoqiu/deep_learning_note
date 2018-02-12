import numpy as np 
import tensorflow as tf 
import pickle
import math
import config as cfg 
import argparse
import random
import tensorflow.contrib.slim as slim

class MLP(object):
	def __init__(self):
		self.layer = cfg.LAYER 
		self.learning_rate = cfg.LEARNING_RATE 
		self.keep_prob = cfg.KEEP_PROB
		self.padding = cfg.PADDING
		self.batch_size = cfg.BATCH_SIZE
		self.epoch = cfg.EPOCH
		self.input = tf.placeholder(shape = [None, (2*self.padding+1)*40], dtype = tf.float32, name = 'inputs')
		self.label = tf.placeholder(shape = [None, 138], dtype = tf.float32, name = 'labels')

	def perceptron_layer(self, idx_layer, input, activation = 'relu'):
		with tf.variable_scope("pl" + str(idx_layer)) as scope:
			# weights = tf.get_variable(shape = [input.shape[1],self.layer[idx_layer-1]], dtype = tf.float32, name = 'weights')
			weights = tf.get_variable(shape = [input.shape[1],self.layer[idx_layer-1]], dtype = tf.float32, initializer = tf.truncated_normal_initializer(), name = 'weights')
			bias    = tf.get_variable(shape = [self.layer[idx_layer - 1]],dtype = tf.float32,  name = 'bias')
		if(activation == 'softmax'):
			return tf.add(tf.matmul(input, weights), bias)
		# return tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(input, weights), bias)), keep_prob = self.keep_prob)
		return tf.layers.batch_normalization(tf.sigmoid(tf.add(tf.matmul(input, weights), bias)))


	# def build_model(self):
	# 	self.net = self.input
	# 	for i in range(1,len(self.layer)):
	# 		self.net = self.perceptron_layer(i, self.net)
	# 	self.net = self.perceptron_layer(len(self.layer), self.net, activation = 'softmax')
	# 	self.sess = tf.Session()
	# 	self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.net, labels = tf.cast(self.label, dtype = tf.float32)))
	# 	# self.optimizer = tf.train.GradientDescentOptimizer(learning_rate = self.learning_rate).minimize(self.loss)
	# 	self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.loss)
	# 	self.init = tf.global_variables_initializer()
	# 	self.saver = tf.train.Saver()
	# 	self.writer = tf.summary.FileWriter('../my_graph', self.sess.graph)
	# 	self.writer.close()
	# 	print("model completed.")

	def build_model(self):
		self.net = self.input
		for i in range(1, len(self.layer)):
			self.net = slim.fully_connected(self.net, self.layer[i-1])
		self.net = slim.fully_connected(self.net, self.layer[-1], activation_fn = None)
		self.sess = tf.Session()
		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.net, labels = tf.cast(self.label, dtype = tf.float32)))
		# self.optimizer = tf.train.GradientDescentOptimizer(learning_rate = self.learning_rate).minimize(self.loss)
		self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.loss)
		self.init = tf.global_variables_initializer()
		self.saver = tf.train.Saver()
		self.writer = tf.summary.FileWriter('../my_graph', self.sess.graph)
		self.writer.close()
		print("model completed.")

	# def preprocess(self, data, label_one_hot):
	# 	new_data = []
	# 	processed_label = []
	# 	for i in range(data.shape[0]):
	# 		new_label = label_one_hot[i][self.padding:-self.padding,:]
	# 		# new_sample = np.zeros([data[i].shape[0]-2*self.padding, data[i].shape[1]*(2*self.padding+1)])
	# 		new_sample = np.lib.pad(data[i], (self.padding,self.padding), mode = 'constant', constant_values = 0)
	# 		for j in range(new_sample.shape[0]):
	# 			new_sample[j,:] = np.reshape(data[i][j:j+(2*self.padding+1), :],(1,-1))
	# 		new_data.append(new_sample)
	# 		processed_label.append(new_label)
	# 	return new_data, processed_label

	def preprocess(self, data, label):
		for i in range(data.shape[0]):
			new_sample = np.pad(data[i], [(self.padding, self.padding),(0,0)], mode = 'constant', constant_values = 0)
			# print(new_sample.shape)
			if(i == 0):
				new_data = new_sample
				new_label = label[i]
			else:
				# new_data = np.append(new_data, new_sample, axis = 0)
				# new_label = np.append(new_label, label[i], axis = 0)
				new_data = np.concatenate((new_data, new_sample), axis = 0)
				new_label = np.concatenate((new_label, label[i]), axis = 0)
			if(i % 100 == 0):
				print(str(i) + "has been concatenated.")

		num = 0
		for i in range(data.shape[0]):
			num += label[i].shape[0]
		table = np.arange(num)
		index = 0
		extra = self.padding
		for i in range(data.shape[0]):
			for j in range(label[i].shape[0]):
				table[index] += extra
				index += 1
			extra += 2*self.padding

		return new_data, new_label, table

	def process(self, data, label):
		len_ori = 0
		for sent in data:
			len_ori += sent.shape[0]
		len_pad = len_ori + self.padding*2*data.shape[0]
		new_data = np.zeros([len_pad, 40])
		# new_label = np.zeros([len_ori, ])
		point_data = 0
		# point_label = 0
		for i in range(data.shape[0]):
			new_sample = np.pad(data[i], [(self.padding, self.padding),(0,0)], mode = 'constant', constant_values = 0)
			new_data[point_data:point_data+new_sample.shape[0],:] = new_sample
			# new_label[point_label:point_label+label[i].shape[0]] = label[i]
			point_data += new_sample.shape[0]
			# point_label += label[i].shape[0]
			if(i % 1000 == 0):
				print(str(i) + 'has been completed.')
		table = np.arange(len_ori)
		index = 0
		extra = self.padding
		for i in range(data.shape[0]):
			for j in range(data[i].shape[0]):
				table[index] += extra
				index += 1
			extra += 2*self.padding
		# return new_data, new_label, table
		return new_data, table

	def one_hot(self, label):
		label = label.astype(int)
		# label_one_hot = []
		# for i_label in label:
		# 	one_label = np.zeros([i_label.shape[0],138])
		# 	for i in range(i_label.shape[0]):
		# 		one_label[i, i_label[i]] = 1
		# 	label_one_hot.append(one_label)
		# label = np.array(label_one_hot)
		new_label = np.zeros([label.shape[0], 138])
		for i in range(label.shape[0]):
			new_label[i, label[i]] = 1
		return new_label



	# def load_data(self, mode):
	# 	if(mode == 'train'):
	# 		data = np.load('../data/dev.npy', encoding = 'latin1')
	# 		label = np.load('../data/dev_labels.npy', encoding = 'latin1')
	# 	elif(mode == 'test'):
	# 		data = np.load('../data/dev.npy', encoding = 'latin1')
	# 		label = np.load('../data/dev_labels.npy', encoding = 'latin1')
	# 	else:
	# 		raise Exception('Wrong mode.')
	# 	label_one_hot = []
	# 	for i_label in label:
	# 		one_label = np.zeros([i_label.shape[0],138])
	# 		for i in range(i_label.shape[0]):
	# 			one_label[i, i_label[i]] = 1
	# 		label_one_hot.append(one_label)
	# 	processed_data, processed_label, idx_table = self.preprocess(data, label_one_hot)

	# 	return processed_data, processed_label, idx_table
		# return data, label_one_hot

	def load_data(self, mode):
		if(mode == 'train'):
			data = np.load('../data/train_padding.npy', encoding = 'latin1')
			label = np.load('../data/train_padding_labels.npy', encoding = 'latin1')
			idx_table = np.load('../data/train_padding_idx.npy', encoding = 'latin1')
			# return data, label, idx_table
		elif(mode == 'test'):
			data = np.load('../data/validation.npy', encoding = 'latin1')
			label = np.load('../data/validation_labels.npy', encoding = 'latin1')
			idx_table = np.load('../data/validation_table.npy', encoding = 'latin1')
		elif(mode == 'validation'):
			data = np.load('../data/validation.npy', encoding = 'latin1')
			label = np.load('../data/validation_labels.npy', encoding = 'latin1')
			idx_table = np.load('../data/validation_idx.npy', encoding = 'latin1')
		else:
			raise Exception('Wrong mode.')
		return data, label, idx_table
		

	# def train(self):
		# data, label = self.load_data('train')
		# self.sess.run(self.init)
		# for epoch in range(10):
		# 	avg_loss = 0
		# 	for idx_batch in range(len(data)):
		# 		data_batch = data[idx_batch]
		# 		label_batch = label[idx_batch]
		# 		_, cost = self.sess.run((self.optimizer, self.loss), feed_dict = {self.input:data_batch, self.label:label_batch})
		# 		avg_loss += cost
		# 	print("epoch %04d : loss = %.9f"%(epoch, avg_loss))
		# self.saver.save(self.sess, '../weights/mlp.ckpt')
		# print('training completed.')

	def train(self):
		data, label, idx_table = self.load_data('train')
		val_data, val_label, val_table = self.load_data('validation') 
		print("load completed.")
		self.sess.run(self.init)
		for epoch in range(self.epoch):
			avg_loss = 0
			shuffle = np.array(random.sample(range(idx_table.shape[0]), idx_table.shape[0]))
			correct_prediction = tf.equal(tf.argmax(self.net, axis = 1), tf.argmax(self.label, axis = 1))
			correct_batch = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
			total_correct = 0
			start = 0
			while(start < shuffle.shape[0]):
				if(start + self.batch_size < shuffle.shape[0]):
					# batch_data = np.zeros(self.batch_size, 40*(2*self.padding+1))
					# batch_label = np.zeros(self.batch_size, 138)
					batch_idx = shuffle[start:start+self.batch_size]
					# print(batch_label.shape)
				else:
					batch_idx = shuffle[start:]
					# batch_data = np.zeros(batch_idx.shape[0], 40*(2*self.padding+1))
					# batch_label = np.zeros(batch_idx.shape[0], 138)
					# print(batch_label.shape)
				# batch_data = data[idx_table[batch_idx]]
				batch_data = np.zeros([batch_idx.shape[0], 40*(2*self.padding+1)])
				for idx in range(batch_idx.shape[0]):
					batch_data[idx,:] = data[idx_table[batch_idx[idx]]-self.padding:idx_table[batch_idx[idx]]+self.padding+1].reshape(1,-1)
				batch_label = label[batch_idx]
				batch_label = self.one_hot(batch_label)
				_, cost, correct = self.sess.run((self.optimizer, self.loss, correct_batch), feed_dict = {self.input:batch_data, self.label:batch_label})
				avg_loss += cost
				total_correct += correct
				start += self.batch_size
			start = 0
			pre_cor = 0
			val_shuffle = np.arange(val_label.shape[0])
			print(val_shuffle.shape[0])
			while(start < val_shuffle.shape[0]):
				if(start + self.batch_size < val_shuffle.shape[0]):
					# batch_data = np.zeros(self.batch_size, 40*(2*self.padding+1))
					# batch_label = np.zeros(self.batch_size, 138)
					batch_idx = val_shuffle[start:start+self.batch_size]
					# print(batch_label.shape)
				else:
					batch_idx = val_shuffle[start:]
					# batch_data = np.zeros(batch_idx.shape[0], 40*(2*self.padding+1))
					# batch_label = np.zeros(batch_idx.shape[0], 138)
					# print(batch_label.shape)
				# batch_data = data[idx_table[batch_idx]]
				batch_data = np.zeros([batch_idx.shape[0], 40*(2*self.padding+1)])
				for idx in range(batch_idx.shape[0]):
					batch_data[idx,:] = val_data[val_table[batch_idx[idx]]-self.padding:val_table[batch_idx[idx]]+self.padding+1].reshape(1,-1)
				batch_label = val_label[batch_idx]
				batch_label = self.one_hot(batch_label)
				corr = self.sess.run(correct_batch, feed_dict = {self.input:batch_data, self.label:batch_label})
				pre_cor += corr
				start += self.batch_size
			print("epoch %04d : loss = %.9f, accuracy = %.9f, val_acc = %.9f"%(epoch, avg_loss, total_correct / idx_table.shape[0], pre_cor / val_label.shape[0]))
		self.saver.save(self.sess, '../weights/mlp.ckpt')


	def predict(self):
		data = np.load("../data/test_padding.npy", encoding = 'latin1')
		table = np.load("../data/test_padding_idx.npy", encoding = 'latin1')
		print("loading completed.")
		self.saver.restore(self.sess, '../weights/weights/mlp.ckpt')
		prediction = tf.argmax(self.net, axis = 1)
		ans = np.zeros([table.shape[0],])
		print(ans.shape)
		start = 0
		point = 0
		while(start < table.shape[0]):
			if(start + self.batch_size < table.shape[0]):
				batch_idx = table[start:start+self.batch_size]
				ans_idx = np.arange(point, point+self.batch_size)
			else:
				batch_idx = table[start:]
				ans_idx = np.arange(point, table.shape[0])
			batch_data = np.zeros([batch_idx.shape[0], 40*(2*self.padding+1)])
			for idx in range(batch_idx.shape[0]):
				batch_data[idx,:] = data[batch_idx[idx] - self.padding:batch_idx[idx]+self.padding+1].reshape(1,-1)
			ans[ans_idx] = self.sess.run(prediction, feed_dict = {self.input:batch_data})
			start += self.batch_size
			point += self.batch_size
		return ans




	def test(self):
		data, label, idx_table = self.load_data('test')
		print("loading completed.")
		self.saver.restore(self.sess, '../weights/mlp.ckpt')
		correct_prediction = tf.equal(tf.argmax(self.net, axis = 1), tf.argmax(self.label, axis = 1))
		correct_batch = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
		total_correct = 0
		start = 0
		while(start < idx_table.shape[0]):
			if(start + self.batch_size < idx_table.shape[0]):
				batch_idx = idx_table[start:start+self.batch_size]
				batch_label = label[start:start+self.batch_size]
			else:
				batch_idx = idx_table[start:]
				batch_label = label[start:]
			batch_data = np.zeros([batch_idx.shape[0], 40*(2*self.padding+1)])
			for idx in range(batch_idx.shape[0]):
				batch_data[idx,:] = data[batch_idx[idx]-self.padding:batch_idx[idx]+self.padding+1].reshape(1,-1)
			# batch_label = label[start:start+self.batch_size]
			total_correct += self.sess.run(correct_batch, feed_dict = {self.input:batch_data, self.label:batch_label})
			start += self.batch_size
		print("accuracy = " + str(total_correct / idx_table.shape[0]))



	def retrain(self):
		data, label, idx_table = self.load_data('train')
		val_data, val_label, val_table = self.load_data('validation') 
		self.saver.restore(self.sess, '../weights/mlp.ckpt')
		for epoch in range(self.epoch):
			avg_loss = 0
			shuffle = np.array(random.sample(range(idx_table.shape[0]), idx_table.shape[0]))
			correct_prediction = tf.equal(tf.argmax(self.net, axis = 1), tf.argmax(self.label, axis = 1))
			correct_batch = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
			total_correct = 0
			start = 0
			while(start < shuffle.shape[0]):
				if(start + self.batch_size < shuffle.shape[0]):
					# batch_data = np.zeros(self.batch_size, 40*(2*self.padding+1))
					# batch_label = np.zeros(self.batch_size, 138)
					batch_idx = shuffle[start:start+self.batch_size]
					# print(batch_label.shape)
				else:
					batch_idx = shuffle[start:]
					# batch_data = np.zeros(batch_idx.shape[0], 40*(2*self.padding+1))
					# batch_label = np.zeros(batch_idx.shape[0], 138)
					# print(batch_label.shape)
				# batch_data = data[idx_table[batch_idx]]
				batch_data = np.zeros([batch_idx.shape[0], 40*(2*self.padding+1)])
				for idx in range(batch_idx.shape[0]):
					batch_data[idx,:] = data[idx_table[batch_idx[idx]]-self.padding:idx_table[batch_idx[idx]]+self.padding+1].reshape(1,-1)
				batch_label = label[batch_idx]
				batch_label = self.one_hot(batch_label)
				_, cost, correct = self.sess.run((self.optimizer, self.loss, correct_batch), feed_dict = {self.input:batch_data, self.label:batch_label})
				avg_loss += cost
				total_correct += correct
				start += self.batch_size
			start = 0
			pre_cor = 0
			val_shuffle = np.arange(val_label.shape[0])
			print(val_shuffle.shape[0])
			while(start < val_shuffle.shape[0]):
				if(start + self.batch_size < val_shuffle.shape[0]):
					# batch_data = np.zeros(self.batch_size, 40*(2*self.padding+1))
					# batch_label = np.zeros(self.batch_size, 138)
					batch_idx = val_shuffle[start:start+self.batch_size]
					# print(batch_label.shape)
				else:
					batch_idx = val_shuffle[start:]
					# batch_data = np.zeros(batch_idx.shape[0], 40*(2*self.padding+1))
					# batch_label = np.zeros(batch_idx.shape[0], 138)
					# print(batch_label.shape)
				# batch_data = data[idx_table[batch_idx]]
				batch_data = np.zeros([batch_idx.shape[0], 40*(2*self.padding+1)])
				for idx in range(batch_idx.shape[0]):
					batch_data[idx,:] = val_data[val_table[batch_idx[idx]]-self.padding:val_table[batch_idx[idx]]+self.padding+1].reshape(1,-1)
				batch_label = val_label[batch_idx]
				batch_label = self.one_hot(batch_label)
				corr = self.sess.run(correct_batch, feed_dict = {self.input:batch_data, self.label:batch_label})
				pre_cor += corr
				start += self.batch_size
			print("epoch %04d : loss = %.9f, accuracy = %.9f, val_acc = %.9f"%(epoch, avg_loss, total_correct / idx_table.shape[0], pre_cor / val_label.shape[0]))
		self.saver.save(self.sess, '../weights/mlp.ckpt')
	

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

