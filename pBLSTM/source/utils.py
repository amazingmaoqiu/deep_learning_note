import numpy as np
import torch


def create_vocab(labels):
    vocab = set()
    for label in labels:
        for char in label:
            vocab.add(char)
    vocab_list = []
    for char in vocab:
        vocab_list.append(char)
    vocab_list.sort()
    return vocab_list

# both input labels & vocab are list
def create_labels(labels, vocab):
	new_labels = []
	for label in labels:
		new_label = []
		for x in label:
			new_label.append(vocab.index(x) + 1)
		# new_label = torch.LongTensor(np.array(new_label))
		new_label = np.array(new_label)
		new_labels.append(new_label)
	return np.array(new_labels)

# input data & labels are both list
def preprocess(batch_data, batch_label=None, is_training=True):
	batch_lengths = torch.LongTensor(list(map(len, batch_data)))
	batch_lengths, perm_idx = batch_lengths.sort(0, descending=True)
	batch_data = batch_data[perm_idx.numpy()]
	# batch_label = batch_label[perm_idx.numpy()]
	if batch_lengths.max() % 8 == 0:
		ideal_length = batch_lengths.max()
	else:
		ideal_length = batch_lengths.max() + (8 - batch_lengths.max() % 8)

	batch_data_tensor = to_variable(torch.zeros(len(batch_data), ideal_length, 40)).float()
	for idx, (seq, seqlen) in enumerate(zip(batch_data, batch_lengths)):
		batch_data_tensor[idx, :seqlen] = torch.FloatTensor(seq)
	if not is_training:
		return batch_data_tensor, batch_lengths
	
	batch_label = batch_label[perm_idx.numpy()]
	batch_label_lengths = torch.LongTensor(list(map(len, batch_label)))
	new_batch_labels = torch.zeros(len(batch_data, ), batch_label_lengths.max()).long()
	for idx, (label, label_length) in enumerate(zip(batch_label, batch_label_lengths)):
		new_batch_labels[idx, :label_length] = torch.LongTensor(label)
	return batch_data_tensor, new_batch_labels, batch_lengths, batch_label_lengths


def to_variable(tensor):
	if torch.cuda.is_available():
		tensor = tensor.cuda()
	return torch.autograd.Variable(tensor)  