import numpy as np
import collections
import sys
import abc
import time
''' 
Data Process only with y's 1 dimension
add extra number -1 in array, let input add one dimension 
'''
class Data():
	def __init__(self, data):
		self.ori_data = np.asarray(data)
		self.label_set = []
		self.label_number = [0, 1]
		self.pre_process_data()
		data_split = round(self.ori_data.shape[0]*2/3)

		self.x = self.ori_data[:,0:-1]
		self.labels = self.ori_data[:,-1]
		self.train_x = self.x[0:data_split]
		self.train_y = self.labels[0:data_split]
		# self.train_x = self.x[:]
		# self.train_y = self.labels[:]
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
		self.label_set = [item for item, count in collections.Counter(labels).items() if count >= 1]
		self.label_set.sort()
		for index, objection in enumerate(self.label_set):
			labels = [index if label==objection else label for label in labels]
		self.label_list = np.arange(len(self.label_set))
		self.ori_data[:,-1] = labels

	'''
	[0, 1]
	[0, 1, 2]
	'''
	def get_label_list(self):
		return self.label_list
	'''
	2
	3
	'''
	def get_label_range(self):
		return len(self.label_list)
	def get_label_index(self, label):
		# print(label)
		# print(self.label_set)
		return self.label_set.index(label)

	def get_x_with_bias(self):
		pass

class BasicActivation():
	def __init__(self):
		pass
	@abc.abstractmethod
	def activate(self, weight_output):
		pass

	@abc.abstractmethod
	def getOutputDeltas(self, except_labels, weight_output):
		pass
	@abc.abstractmethod
	def getHiddenDeltas(self, deltasK, outputY, weightK):
		pass

class Sigmoid(BasicActivation):
	def __init__(self):
		pass
	def activate(self, weight_output):
		return 1 / (1 + np.exp(-weight_output))

	def getOutputDeltas(self, except_labels, weight_output):
		return (except_labels - weight_output) * weight_output * (1 - weight_output)
	
	def getHiddenDeltas(self, deltasK, weight_output, weightK): 
		weightK = np.copy( np.delete(weightK, weightK.shape[1]-1, axis=1) )
		sum_deltasK_dot_weightK = np.sum(weightK*deltasK, axis=0).reshape((-1, 1))
		return weight_output * (1 - weight_output) * sum_deltasK_dot_weightK

class ReLU(BasicActivation):
	def __init__(self):
		pass
	def activate(self, weight_output):
		# print(weight_output)
		# time.sleep(0.001)
		# print(np.maximum(0, weight_output))
		return np.maximum(0, weight_output)

	def getOutputDeltas(self, except_labels, weight_output):
		# print(weight_output)
		except_labels = except_labels - weight_output
		# print(except_labels)
		except_labels[weight_output<0] = 0
		return except_labels
	
	def getHiddenDeltas(self, deltasK, weight_output, weightK): 
		weightK = np.copy( np.delete(weightK, weightK.shape[1]-1, axis=1) )
		sum_deltasK_dot_weightK = np.sum(weightK*deltasK, axis=0).reshape((-1, 1))

		sum_deltasK_dot_weightK[weight_output<0] = 0
		# print(sum_deltasK_dot_weightK)
		return sum_deltasK_dot_weightK


def sign(weight_output):
	return list(map(lambda label: 1 if label > 0 else 0, weight_output.flatten()))

def sigmoid(weight_output):
	return 1 / (1 + np.exp(-weight_output))
	# return np.asarray( list(map(lambda result: 1 / (1 + np.exp(-result)), weight_output.flatten())) )

def relu(weight_output):
	return np.maximum(0, weight_output)

