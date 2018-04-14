import numpy as np
import torch
from model import *
from utils import create_vocab, create_labels, preprocess, to_variable
from random import shuffle
import config as cfg
from os.path import isfile

def train():
	data = np.load("../dataset/dev.npy")
	labels = np.load("../dataset/dev_transcripts.npy")


	# temorary dataset
	data = data[0:2]
	labels = labels[0:2]
	# temporary dataset



	vocab = create_vocab(labels)

	labels = create_labels(labels, vocab)
	
	shuffle_index = np.arange(len(data))
	shuffle(shuffle_index)

	batch_size = cfg.BATCH_SIZE
	learning_rate = cfg.LEARNING_RATE

	# my_listener = Listener(40, 256, 0.0)
	# my_speller  = Speller(33, 512, 512, 256, 3)

	if isfile("../weights/listener.pt"):
		with open("../weights/listener.pt", 'rb') as fl:
			my_listener = torch.load(fl)
		with open("../weights/speller.pt", 'rb') as fs:
			my_speller = torch.load(fs)
		print("model loading completed.")
	else:
		my_listener = Listener(40, 256, 0.0)
		my_speller  = Speller(33, 512, 512, 256, 3)

	loss_fn = torch.nn.CrossEntropyLoss(reduce=False)
	my_optimizer = torch.optim.Adam([{'params':my_speller.parameters()}, {'params':my_listener.parameters()}], lr=cfg.LEARNING_RATE)

	start_index = 0
	for epoch in range(cfg.EPOCH):
		losses = 0.0
		start_index = 0
		while(start_index + batch_size <= len(data)):
			batch_data = data[shuffle_index[start_index:start_index + batch_size]]
			batch_labels = labels[shuffle_index[start_index:start_index + batch_size]]
			batch_data, batch_labels, batch_lengths, batch_label_lengths = preprocess(batch_data, batch_labels)
			one_hot_batch_labels = OneHot(batch_labels, 33)
			listener_output = my_listener(batch_data, batch_lengths)
	
			speller_output = my_speller(batch_labels.size(1), listener_output, one_hot_batch_labels)
	
			batch_loss = loss_fn(speller_output[0].contiguous().view(-1, 33), torch.autograd.Variable(batch_labels).view(-1,))
			batch_loss = batch_loss.view(speller_output[0].size(0), speller_output[0].size(1))
			mask = torch.zeros(batch_loss.size())
			for i in range(batch_label_lengths.size(0)):
				mask[i, :batch_label_lengths[i]] = 1.0
			batch_loss = torch.mul(batch_loss, torch.autograd.Variable(mask))
			batch_loss = torch.sum(batch_loss) / torch.sum(mask)
			print("epoch {} batch_loss == {:.5f}".format(epoch, batch_loss.data[0]))
			batch_loss.backward()
			losses += batch_loss.data.cpu().numpy()
			my_optimizer.step()
	
			start_index += batch_size
			# break
		if(epoch % 3 == 0):
			with open("../weights/listener.pt", 'wb') as fl:
				torch.save(my_listener, fl)
			with open("../weights/speller.pt", 'wb') as fs:
				torch.save(my_speller, fs)


if __name__ == "__main__":
	train()
	print("completed!")