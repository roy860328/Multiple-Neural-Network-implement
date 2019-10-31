import numpy as np

from . import basic
from neuralnetwork.neurons import neurons as ns
from utils import utils


class MLP(basic.Basic):

	def __init__(self, *page, data, hidden_neurons=[1], initial_learning_rate=0.8, max_epoches=10, least_error_rate=None, mode="max_epoches"):
		super().__init__(data, initial_learning_rate, max_epoches, least_error_rate, mode, *page)
		self.page = page[0]
		self.page.page_component.print_to_result("Init: MLP")
		self._initial_neurons(hidden_neurons)
		self.weights[0].weight = np.asarray([[1, 1, -1.2], [1, 1, 0.3]])
		self.weights[1].weight = np.asarray([[0.4, 0.8, 0.5]])

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
		network_architecture.append((1, dim))
		
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
		outputY = np.reshape(outputY, (-1, 1))
		self._print("\n\nexceptY", exceptY)
		# self._print("except_labels", except_labels)
		self._print("outputY", outputY)
		weight.deltas = (exceptY - outputY) * outputY * (1 - outputY)
		self._print("weight.deltas", weight.deltas)

	def _back_propagation_hidden_layer(self, deltasK, outputY, weightK, weightJ):
		### 對weightK降維，因為weightJ不需要更新bias(weightJ輸出Y的維度有多1維bias為-1)
		self._print("\noutputY", outputY)
		self._print("weightK", weightK)
		weightK = np.copy( np.delete(weightK, weightK.shape[1]-1, axis=1) )
		# self._print("weightK", weightK)
		sum_deltasK_dot_weightK = np.sum(weightK*deltasK, axis=0).reshape((-1, 1))
		weightJ.deltas = outputY * (1 - outputY) * sum_deltasK_dot_weightK
		self._print("weightJ.deltas", weightJ.deltas)

	def _update_weight(self, intputX):
		if(intputX.ndim == 1):
			intputX = np.reshape(intputX, (1, -1))
		intputX = np.insert(intputX, intputX.shape[1], -1, axis=1).T
		for index, weight in enumerate(self.weights):
			if(index == 0):
				weight.weight = weight.weight + self.learning_rate * weight.deltas * intputX.T
				self._print("\n\nweight.weight", weight.weight)
			else:
				weightI.result = np.insert(weightI.result, weightI.result.shape[0], -1, axis=0)
				self._print("weightI.result", weightI.result)
				weight.weight = weight.weight + self.learning_rate * weight.deltas * weightI.result.T
				self._print("weight.weight", weight.weight)
			weightI = weight

	def _pass_activation_function(self, weight_output):
		return utils.sigmoid(weight_output)

	def _cal_correct_rate(self, datasetX, datasetY):
		super(MLP, self)._cal_correct_rate(datasetX, datasetY)

		result = self._forward_propagation(datasetX)
		self._print("\n\nresult", result)
		result = result>0.5
		result = result[0]
		self._print("result", result)
		self._print("datasetY", datasetY)
		correct_n = sum(result == datasetY)
		correct_rate = np.round(correct_n/datasetY.shape[0]*100, 4)
		return result, correct_n, correct_rate