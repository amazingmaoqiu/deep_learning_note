import numpy as np 
import torch
from model import *
from utils import *

def predict():
	data = np.load("../dataset/dev.npy")

	data = data[0:2]

	with open("../weights/listener.pt", 'rb') as fl:
		my_listener = torch.load(fl)
	with open("../weights/speller.pt", 'rb') as fs:
		my_speller = torch.load(fs)
	print("model loading completed.")

	predictions = []
	for batch_data in data:
		prediction = []
		batch_data, batch_lengths = preprocess(np.array([batch_data]), is_training=False)
		listener_output = my_listener(batch_data, batch_lengths)
		one_hot_batch_label = OneHot(torch.LongTensor(np.zeros([1, 1])), 33)
		for step in range(listener_output.size(1)):
			speller_output = my_speller(1, listener_output[:, 0:step+1, :], one_hot_batch_label)
			predict = speller_output[0].squeeze().data.numpy()
			predict = np.argmax(predict)
			prediction.append(predict)
			one_hot_batch_label = OneHot(torch.LongTensor(np.array([[predict]])), 33)
			# break
		print(prediction)
		predictions.append(prediction)

		




if __name__ == "__main__":
	predict()