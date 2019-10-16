import abc
import threading 
import numpy as np
from utils.utils import Data 
from neuralnetwork.neurons import neurons as ns

class Basic(abc.ABC, threading.Thread):
	def __init__(self, data, initial_learning_rate, total_epoches, least_error_rate, mode, *page):
		super().__init__()
		threading.Thread.__init__(self)
		self.page = page[0]
		self.data = data
		self.learning_rate = initial_learning_rate
		self._total_epoches = total_epoches
		self._least_error_rate = least_error_rate
		self.mode = mode

		self.current_iterations = 0
		self.stop_thread = False
		self._correct_rate = 0
		self._epoches_print_rate = max(1, self._total_epoches//10)

		self._print_init_set()

	def run(self):
		self.page.page_component.print_to_result("\n=== Thread.start() ===")
		self.start_training()
		self.page.page_component.print_to_result("\n=== Calculate correct rate ===")
		self.cal_correct_rate("Training", self.data.train_x, self.data.train_y)
		self.cal_correct_rate("Testing", self.data.test_x, self.data.test_y)
		result = self.cal_correct_rate("All", self.data.x, self.data.labels)
		inputX = self.data.x[:,0:-1]
		self.page.graph.draw_result(inputX, result, self.weights)
		
		self.page.finish_training()
		self._print_weights()

	''' Main Training Loop '''
	def start_training(self):
		self.page.page_component.print_to_result("\n=== Start to train ===")
		self.stop_thread = False

		''' Main Training Loop '''
		for self.current_iterations in range(self._total_epoches):
			## User stop process
			if(self.stop_thread): 
				self.page.page_component.print_to_result("Interrupt training ...")
				return
			## Traing Step Alg
			self._train_step()
			## print current Epoches state
			if(self.current_iterations%self._epoches_print_rate == 0):
				self.page.page_component.print_to_result("Epoches: {}".format(self.current_iterations))

		self.page.page_component.print_to_result("\n=== Training Finish ===")

	def stop_training(self):
		self.stop_thread = True

	""" compare current output with data """
	def cal_correct_rate(self, dataformat, datasetX, datasetY):
		assert datasetX.shape[0] == datasetY.shape[0], "Error: datasetX&Y size not same"
		if (len(datasetX) == 0): return "Error: DatasetX&Y are Null", 0, 0

		result = self._forward_propagation(datasetX)
		correct_n = sum(result == datasetY)
		correct_rate = round(correct_n/datasetY.shape[0]*100, 4)

		self.page.page_component.print_to_result("{} Data correct rate: {}/{}, {}%".format(dataformat, correct_n, datasetY.shape[0], correct_rate))
		return result

	def _train_step(self):
		for index in range(len(self.data.train_x)):
			self._forward_propagation(self.data.train_x[index])
			self._adjust_weight(self.data.train_x[index], self.data.train_y[index])

	def _forward_propagation(self, inputX):
		for weight in self.weights:
			weight.result = weight.weight @ inputX.T
			weight.result = self._pass_activation_function(weight.result)
			inputX = weight.result
		return inputX

	def _print_init_set(self):
		self.page.page_component.print_to_result("\n=== Init: pass para to basic neural network ===")
		self.page.page_component.print_to_result("Data: {}".format(self.data))
		self.page.page_component.print_to_result("Learning rate: {}".format(self.learning_rate))
		self.page.page_component.print_to_result("Totale epoches: {}".format(self._total_epoches))
		self.page.page_component.print_to_result("Mode: {}".format(self.mode))
		self.page.page_component.print_to_result("Epoches print rate: {}".format(self._epoches_print_rate))

	def _print_weights(self):
		self.page.page_component.print_to_result("\n=== Weights ===")
		for weight in self.weights:
			self.page.page_component.print_to_result("{}".format(weight))

	@abc.abstractmethod
	def _initial_neurons(self, hidden_layers):
		"""  """
	@abc.abstractmethod
	def _adjust_weight(self, intputX, outputY):
		""" each train step with the training algorithm """

	@abc.abstractmethod
	def _pass_activation_function(self, weight_output):
		""" pass neuron """

	@property
	def data(self):
		return self._data

	@data.setter
	def data(self, data):
		self.page.page_component.print_to_result("\n=== Load dataset ===")
		self._data = Data(data)