import numpy as np
import collections
import sys

from . import basic
from neuralnetwork.neurons import neurons as ns
from utils import utils


class MLP(basic.Basic):

	def __init__(self, *page, data, hidden_neurons=[1], initial_learning_rate=0.8, max_epoches=10, least_correct_rate=None, mode="max_epoches"):
		super().__init__(data, initial_learning_rate, max_epoches, least_correct_rate, mode, *page)
		self.page = page[0]
		self.page.page_component.print_to_result("Init: MLP")
		self.activation = utils.ReLU()
		self.activation = utils.Sigmoid()
		self._initial_neurons(hidden_neurons)


	''' Call by thread '''
	def run(self):
		super(MLP, self).run()
		

	def _initial_neurons(self, hidden_layers):
		dim = self.data.x.shape[1] + 1
		network_architecture = []
		''' hidden layers '''
		for layer_number in hidden_layers:
			network_architecture.append((layer_number, dim))
			dim = layer_number + 1
		''' output layers '''
		network_architecture.append((len(self.data.get_label_list()), dim))
		
		super(MLP, self)._initial_neurons(network_architecture)
		

	def start_training(self): 
		super(MLP, self).start_training()
		try: 
			pass
		finally: 
			pass

	''' 
	layer order
	i, j, k 
	'''
	def _adjust_weight(self, intputX, exceptY):
		self._cal_deltas(exceptY)
		self._update_weight(intputX)

	''' 
	weight:
	w = w + n * deltasJ * j

	Exception gradient:
	deltasK = (except - outputY)*outputY*(1-outputY)

	Hidden layer gradient:
	deltas = y(1-y)E(deltasK*weightK)
	'''
	def _cal_deltas(self, exceptY):
		for index, weight in enumerate(reversed(self.weights)):
			if(index == 0):
				self._back_propagation_output_layer(exceptY, weight.result, weight)
			else: 
				self._back_propagation_hidden_layer(weightK.deltas, weight.result, weightK.weight, weight)
			weightK = weight
	def _back_propagation_output_layer(self, exceptY, outputY, weight):
		except_labels = np.zeros((len(self.data.get_label_list()), 1))
		except_labels[int(exceptY)] = 1
		outputY = np.reshape(outputY, (-1, 1))
		# self._print("\n\nexceptY", exceptY)
		# self._print("except_labels", except_labels)
		# self._print("outputY", outputY)
		weight.deltas = self.activation.getOutputDeltas(except_labels, outputY)
		# print(except_labels)

	def _back_propagation_hidden_layer(self, deltasK, outputY, weightK, weightJ):
		### 對weightK降維，因為weightJ不需要更新bias(weightJ輸出Y的維度有多1維bias為-1)
		# self._print("weightK", weightK)
		weightJ.deltas = self.activation.getHiddenDeltas(deltasK, outputY, weightK)
		# print(weightJ.deltas)

	def _update_weight(self, intputX):
		if(intputX.ndim == 1):
			intputX = np.reshape(intputX, (1, -1))
		intputX = np.insert(intputX, intputX.shape[1], -1, axis=1).T
		for index, weight in enumerate(self.weights):
			if(index == 0):
				weight.weight = weight.weight + self.learning_rate * weight.deltas * intputX.T
			else:
				weightI.result = np.insert(weightI.result, weightI.result.shape[0], -1, axis=0)
				weight.weight = weight.weight + self.learning_rate * weight.deltas * weightI.result.T
			weightI = weight

	def _pass_activation_function(self, weight_output):
		# print( weight_output)
		# print (np.maximum(0, weight_output))
		return self.activation.activate(weight_output)
		# return utils.sigmoid(weight_output)

	def _cal_correct_rate(self, datasetX, datasetY):
		super(MLP, self)._cal_correct_rate(datasetX, datasetY)
		# self.page.page_component.print_to_result(self.weights)
		
		result = self._forward_propagation(datasetX)
		RMSE = self._cal_RMSE(result, datasetY)
		# print(datasetY)
		# print(result)
		# print(result.shape)
		result[result > 0.5] = 1
		result[result < 0.5] = 0
		output_transform = self._transform_output_to_label(result)
		# print(output_transform)
		# self._print("output_transform", output_transform)
		# self._print("result", result)
		# self._print("datasetY", datasetY)

		is_same = (output_transform == datasetY)
		correct_n = sum(is_same)
		correct_rate = round(correct_n/datasetY.shape[0]*100, 4)
		return output_transform, correct_n, correct_rate, RMSE

	def _cal_RMSE(self, output, datasetY):
		RMSE = 0
		datasetY = list(map(int, datasetY))
		one_hot_labels = np.zeros( (self.data.get_label_range(), len(datasetY)) )
		for index in range(one_hot_labels.shape[1]):
			one_hot_labels[datasetY[index], index] = 1

		# print(one_hot_labels)
		RMSE = np.sum((one_hot_labels-output)**2)/one_hot_labels.shape[1]/2
		# print(RMSE)
		return RMSE

	def _transform_output_to_label(self, result):
		output_transform = np.zeros(result.shape[1])
		for index in range(result.shape[1]):
			count = collections.Counter(result[:, index])
			if count[1] > 1 or count[1] < 1:
				output_transform[index] = -1
			else:
				find_label_number = np.where(result[:, index] == np.amax(result[:, index]))[0]
				output_transform[index] = find_label_number
		return output_transform