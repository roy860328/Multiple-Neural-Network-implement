import numpy as np

''' 
Data Process only with y's 1 dimension
add extra number -1 in array, let input add one dimension 
'''
class Data():
	def __init__(self, data):
		self.ori_data = np.asarray(data)
		np.random.shuffle(self.ori_data)
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
