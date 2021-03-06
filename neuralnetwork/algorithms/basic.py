import abc
import threading 
import numpy as np
import sys

from utils.utils import Data 
from neuralnetwork.neurons import neurons as ns

# class Basic(abc.ABC, threading.Thread):
class Basic(abc.ABC):
	def __init__(self, *page):
		super().__init__()
		self.page = page[0]

	@abc.abstractmethod
	def run(self):
		"""  """

	@abc.abstractmethod
	def start_training(self):
		""" Main Training Loop """
	
	@abc.abstractmethod
	def stop_training(self):
		"""  """
	
	@abc.abstractmethod
	def _initial_neurons(self, network_architecture):
		"""  """
	
	@abc.abstractmethod
	def _train_step(self):
		"""  """
	
	@abc.abstractmethod
	def _cal_correct_rate(self, dataformat, datasetX, datasetY):
		""" compare current output with data """
		assert datasetX.shape[0] == datasetY.shape[0], "Error: datasetX&Y size not same"
		if (len(datasetX) == 0): return "Error: DatasetX&Y are Null", 0, 0
		self.page.page_component.print_to_result("{}".format(dataformat))

	@abc.abstractmethod
	def _adjust_weight(self, intputX, outputY):
		""" each train step with the training algorithm """

	@abc.abstractmethod
	def _pass_activation_function(self, weight_output):
		""" pass neuron """
	
	@abc.abstractmethod
	def _print_init_set(self):
		self.page.page_component.print_to_result("\n=== Init: pass para to basic neural network ===")
	


class BasicPerceptron(Basic):
	def __init__(self, data, initial_learning_rate, total_epoches, least_correct_rate, mode, *page):
		super().__init__(*page)
		# threading.Thread.__init__(self)
		self.data = data
		self.learning_rate = initial_learning_rate
		self._total_epoches = total_epoches
		self._least_correct_rate = least_correct_rate
		self.mode = mode

		self.current_iterations = 0
		self.stop_thread = False
		self._correct_rate = 0
		self._epoches_print_rate = max(1, self._total_epoches//10)
		self.RMSE = 0

		self._print_init_set()

	def run(self):
		## Traing
		self.page.page_component.print_to_result("\n=== Thread.start() ===")
		self.start_training()
		## Test result
		self.page.page_component.print_to_result("\n=== Calculate correct rate ===")
		_, _      = self._cal_correct_rate("\nTraining", self.data.train_x, self.data.train_y)
		_, _      = self._cal_correct_rate("Testing", self.data.test_x, self.data.test_y)
		result, _ = self._cal_correct_rate("\nAll", self.data.x, self.data.labels)
		## Draw to canvas
		self.page.page_component.print_to_result("\n=== Draw esult ===")
		self.page.graph.draw_result(self.data.x, result, self.weights)
		## Print 
		self._print_weights()
		self.page.finish_training()

	''' Main Training Loop '''
	def start_training(self):
		self.page.page_component.print_to_result("\n=== Start to train ===")
		self.stop_thread = False

		''' Main Training Loop '''
		while self.current_iterations < self._total_epoches or self._least_correct_rate > current_correct_rate:
			## User stop process
			if(self.stop_thread): 
				self.page.page_component.print_to_result("Interrupt training ...")
				return
			## Traing Step Alg
			self._train_step()
			## print current Epoches state
			if(self.current_iterations%self._epoches_print_rate == 0):
				self.page.page_component.print_to_result("\nEpoches: {}".format(self.current_iterations))
				_, current_correct_rate = self._cal_correct_rate("Training", self.data.train_x, self.data.train_y)
				result, _ 				= self._cal_correct_rate("Testing", self.data.test_x, self.data.test_y)
				self.page.graph.draw_result(self.data.test_x, result, self.weights)
			self.current_iterations += 1

		self.page.page_component.print_to_result("\n=== Training Finish ===")

	def stop_training(self):
		self.stop_thread = True


	def _initial_neurons(self, network_architecture):
		self.weights = ns.LayersImplement().create_neurons_layer(network_architecture)
		self.page.page_component.print_to_result("\n=== Init weights ===")
		self.page.page_component.print_to_result(network_architecture)
		self.page.page_component.print_to_result(self.weights)
	
	def _train_step(self):
		for index in range(len(self.data.train_x)):
			self._forward_propagation(self.data.train_x[index])
			self._adjust_weight(self.data.train_x[index], self.data.train_y[index])

	def _forward_propagation(self, intputX):
		if(intputX.ndim == 1):
			intputX = np.reshape(intputX, (1, -1))
		intputX = intputX.T
		# self._print("intputX", intputX.shape)
		for weight in self.weights:
			intputX = np.insert(intputX, intputX.shape[0], -1, axis=0)
			# print(intputX)
			weight.result = weight.weight @ intputX
			# print(weight.weight)
			# print(weight.result)
			weight.result = self._pass_activation_function(weight.result)
			# print(weight.result)
			intputX = weight.result
		# print(intputX)
		return intputX

	""" compare current output with data """
	def _cal_correct_rate(self, dataformat, datasetX, datasetY):
		super(BasicPerceptron, self)._cal_correct_rate()
		result, correct_n, correct_rate, self.RMSE = self.cal_correct_rate(datasetX, datasetY)

		self.page.page_component.print_to_result("Data correct rate: {}/{}, {}%  \nError rate                : {}%  \nRMSE                      : {}"\
													 .format(correct_n, datasetY.shape[0], correct_rate, 100-round(correct_rate, 3), round(self.RMSE, 10)) )
		# self.page.page_component.print_to_result(self.weights)
		return result, correct_rate

	def _adjust_weight(self, intputX, outputY):
		""" each train step with the training algorithm """
		super(BasicPerceptron, self)._adjust_weight()

	def _pass_activation_function(self, weight_output):
		""" pass neuron """
		super(BasicPerceptron, self)._pass_activation_function()

	@abc.abstractmethod
	def cal_correct_rate(self, datasetX, datasetY):
		""" _cal_correct_rate """

	@property
	def data(self):
		return self._data

	@data.setter
	def data(self, data):
		self.page.page_component.print_to_result("\n=== Load dataset ===")
		self._data = Data(data)


	def _print_init_set(self):
		super(BasicPerceptron, self)._print_init_set()

		self.page.page_component.print_to_result("Data: {}".format(self.data))
		self.page.page_component.print_to_result("Learning rate: {}".format(self.learning_rate))
		self.page.page_component.print_to_result("Totale epoches: {}".format(self._total_epoches))
		self.page.page_component.print_to_result("Mode: {}".format(self.mode))
		self.page.page_component.print_to_result("Epoches print rate: {}".format(self._epoches_print_rate))

	def _print_weights(self):
		self.page.page_component.print_to_result("\n=== Weights ===")
		for weight in self.weights:
			self.page.page_component.print_to_result("{}".format(weight))

	def _print(self, string, input_obj):
		print("{}: {}".format(string, input_obj))