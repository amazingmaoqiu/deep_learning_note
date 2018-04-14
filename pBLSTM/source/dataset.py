import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class My_Dataset(Dataset):
	def __init__(self, data, labels):
		self.data = data
		self.labels = labels
		self.SequenceLength = []
		for dd in data:
			self.SequenceLength.append(dd.shape[0])

	def __getitem__(self, index):
		return self.data[index], self.labels[index], self.SequenceLength[index]

	def __len__(self):
		return len(data)

