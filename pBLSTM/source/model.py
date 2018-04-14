import numpy as np
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn as nn


class pBLSTM(nn.Module):
	def __init__(self, input_dim, hidden_dim, dropout_rate=0.0):
		super(pBLSTM, self).__init__()
		self.pblstm = nn.LSTM(input_dim*2, hidden_dim, 1, bidirectional=True, dropout=dropout_rate, batch_first=True)

	def forward(self, input_data, input_lengths):
		batch_size = input_data.size(0)
		time_step = input_data.size(1)
		feature_dim = input_data.size(2)
		input_data = input_data.contiguous().view(batch_size, time_step // 2, feature_dim*2)

		input_lengths = (input_lengths - 1) / 2 + 1

		lengths = []
		for length in input_lengths:
			lengths.append(length)
		lengths[0] = input_data.size(1)

		input_data = pack_padded_sequence(input_data, lengths, batch_first=True)
		output, _ = self.pblstm(input_data)
		output, _ = pad_packed_sequence(output, batch_first=True)

		return output, input_lengths


class Listener(nn.Module):
	def __init__(self, input_dim, hidden_dim, dropout_rate):
		super(Listener, self).__init__()
		self.pBLSTM1 = pBLSTM(input_dim, hidden_dim, dropout_rate=dropout_rate)
		self.pBLSTM2 = pBLSTM(2*hidden_dim, hidden_dim, dropout_rate=dropout_rate)
		self.pBLSTM3 = pBLSTM(2*hidden_dim, hidden_dim, dropout_rate=dropout_rate)

	def forward(self, input_data, input_lengths):
		output, input_lengths = self.pBLSTM1(input_data, input_lengths)
		output, input_lengths = self.pBLSTM2(output, input_lengths)
		output, input_lengths = self.pBLSTM3(output, input_lengths)
		return output


class Speller(nn.Module):
	def __init__(self, output_class_dim, speller_hidden_dim, output_size_attention, listener_hidden_dim, num_layers):
		super(Speller, self).__init__()
		self.label_dim = output_class_dim
		self.rnn = nn.LSTM(output_class_dim+speller_hidden_dim, speller_hidden_dim, num_layers=num_layers)
		self.attention = Attention(2*listener_hidden_dim, output_size_attention)
		self.prediction_distribution = nn.Linear(speller_hidden_dim*2, output_class_dim)
		self.softmax = nn.LogSoftmax(dim=-1)

	def forward_step(self, input_data, last_hidden_state, listener_output):
		output, hidden_state = self.rnn(input_data, last_hidden_state)
		attention_score, context = self.attention(output, listener_output)
		concate_output = torch.cat([output.squeeze(dim=1), context], dim=-1)
		logits = self.prediction_distribution(concate_output)
		prediction = self.softmax(logits)
		return prediction, hidden_state, context, attention_score

	def forward(self, max_time_step, listener_output, label=None):
		batch_size = listener_output.size(0)
		output_word = OneHot(torch.FloatTensor(np.zeros([batch_size, 1])), encoding_dim=self.label_dim)
		rnn_input = torch.cat([output_word, listener_output[:, 0:1, :]], dim=-1) 
		hidden_state = None
		predictions = []
		output_sequence = []
		attention_record = []
		for step in range(max_time_step):
			prediction, hidden_state, context, attention_score = self.forward_step(rnn_input, hidden_state, listener_output)
			predictions.append(prediction)
			attention_record.append(attention_score)
			if label is not None:
				output_word = label[:, step:step+1, :].float()
			else:
				output_word = prediction.unsqueeze(1)
			rnn_input = torch.cat([output_word, context.unsqueeze(1)], dim=-1)
		return torch.stack(predictions).transpose(0, 1), attention_record


class Attention(nn.Module):
	def __init__(self, input_size, output_size):
		super(Attention, self).__init__()
		self.input_size = input_size
		self.output_size = output_size
		self.relu = nn.ReLU()
		self.softmax = nn.Softmax(dim=-1)
		self.linear1 = nn.Linear(self.input_size, self.output_size)
		self.linear2 = nn.Linear(self.input_size, self.output_size)
		self.activate = nn.ReLU()

	def forward(self, decoder_output, listener_output):
		decoder_output = self.relu(self.linear1(decoder_output))
		listener_output = self.relu(self.linear2(listener_output))
		energy = torch.bmm(decoder_output, listener_output.transpose(1,2)).squeeze(dim=1)
		attention_score = self.softmax(energy)
		context = torch.sum(listener_output*attention_score.unsqueeze(2).repeat(1,1,listener_output.size(2)),dim=1)
		return attention_score, context


def OneHot(input_x, encoding_dim=63):
    if type(input_x) is torch.autograd.Variable:
        input_x = input_x.data 
    input_type = type(input_x)
    batch_size = input_x.size(0)
    time_steps = input_x.size(1)
    input_x = input_x.unsqueeze(2).type(torch.LongTensor)
    onehot_x = torch.autograd.Variable(torch.LongTensor(batch_size, time_steps, encoding_dim).zero_().scatter_(-1,input_x,1)).type(input_type)
    
    return onehot_x