import numpy as np
import collections

''' 
Data Process only with y's 1 dimension
add extra number -1 in array, let input add one dimension 
'''
class Data():
	def __init__(self, data):
		self.ori_data = np.asarray(data)
		self.label_number = [0, 1]
		self.pre_process_data()
		data_split = round(self.ori_data.shape[0]*2/3)

		self.x = np.insert(self.ori_data, -1, -1, axis=1)
		self.x = self.x[:,0:-1]
		self.labels = self.ori_data[:,-1]
		self.train_x = self.x[0:data_split]
		self.train_y = self.labels[0:data_split]
		self.test_x = self.x[data_split:]
		self.test_y = self.labels[data_split:]

	def __call__(self):
		return self.x

	def __str__(self):
		return '{}'.format(self.ori_data.shape)

	def pre_process_data(self):
		np.random.shuffle(self.ori_data)
		self.flatten_lable()
	''' 
	before: 1, 2 label
	after: 0, 1 label
	'''
	def flatten_lable(self):
		labels = self.ori_data[:,-1]
		self.label_number = [item for item, count in collections.Counter(labels).items() if count >= 1]
		self.label_number.sort()
		for index, objection in enumerate(self.label_number):
			labels = [index if label==objection else label for label in labels]
		self.label_number = np.arange(len(self.label_number))
		self.ori_data[:,-1] = labels
		# print(self.ori_data[:,-1])

def sign(weight_output):
	return list(map(lambda label: 1 if label > 0 else 0, weight_output.flatten()))

def sigmoid(weight_output):
	return np.asarray( list(map(lambda result: 1 / (1 + np.exp(-result)), weight_output.flatten())) )