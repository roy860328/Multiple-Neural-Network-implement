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

	''' Call by thread '''
	def run(self):
		super(MLP, self).run()
		

	def _initial_neurons(self, hidden_layers):
		super(MLP, self)._initial_neurons(hidden_layers)
		dim = self.data.ori_data.shape[1]
		network_architecture = []
		''' hidden layers '''
		for layer_number in hidden_layers:
			network_architecture.append((layer_number, dim))
			dim = layer_number
		''' output layers '''
		network_architecture.append((len(self.data.label_number), dim))
		print(network_architecture)
		self.weights = ns.LayersImplement().create_neurons_layer(network_architecture)
		self.page.page_component.print_to_result("\n=== Init weights ===")
		self.page.page_component.print_to_result(self.weights)

	def start_training(self): 
		super(MLP, self).start_training()
		try: 
			pass
		finally: 
			pass

	def _adjust_weight(self, intputX, exceptY):
		if self.weights[0].result[0] == 0 and exceptY != 0:
			self.weights[0].weight = self.weights[0].weight + self.learning_rate * intputX
		elif self.weights[0].result[0] == 1 and exceptY != 1:
			self.weights[0].weight = self.weights[0].weight - self.learning_rate * intputX

	def _pass_activation_function(self, weight_output):
		return utils.sigmoid(weight_output)