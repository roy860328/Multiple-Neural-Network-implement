import numpy as np

from . import basic
from neuralnetwork.neurons import neurons as ns
from utils import utils 


class SimplePerceptron(basic.Basic):

	def __init__(self, *page, data, hidden_neurons=[1], initial_learning_rate=0.8, max_epoches=10, least_error_rate=None, mode="max_epoches"):
		hidden_neurons = [1]
		super().__init__(data, initial_learning_rate, max_epoches, least_error_rate, mode, *page)
		self.page = page[0]
		self.page.page_component.print_to_result("Init: SimplePerceptron")
		self._initial_neurons(hidden_neurons)

	''' Call by thread '''
	def run(self):
		super(SimplePerceptron, self).run()

	def _initial_neurons(self, hidden_layers):
		dim = self.data.x.shape[1] + 1
		network_architecture = []
		''' output layers '''
		network_architecture.append((hidden_layers[0], dim))
		
		super(SimplePerceptron, self)._initial_neurons(network_architecture)

	def start_training(self): 
		super(SimplePerceptron, self).start_training()
		try: 
			pass
		finally: 
			pass

	def _adjust_weight(self, intputX, exceptY):
		intputX = np.insert(intputX, len(intputX), -1)
		# print(intputX)
		if self.weights[0].result[0] == 0 and exceptY != 0:
			self.weights[0].weight = self.weights[0].weight + self.learning_rate * intputX
		elif self.weights[0].result[0] == 1 and exceptY != 1:
			self.weights[0].weight = self.weights[0].weight - self.learning_rate * intputX

	def _pass_activation_function(self, weight_output):
		return utils.sign(weight_output)

	def _cal_correct_rate(self, datasetX, datasetY):
		super(SimplePerceptron, self)._cal_correct_rate(datasetX, datasetY)
		
		result = self._forward_propagation(datasetX)
		correct_n = sum(result == datasetY)
		correct_rate = round(correct_n/datasetY.shape[0]*100, 4)
		return result, correct_n, correct_rate