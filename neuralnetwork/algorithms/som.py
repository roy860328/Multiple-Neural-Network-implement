import numpy as np
import collections
import sys
import math  

from . import basic
from neuralnetwork.neurons import neurons as ns
from utils.utils import *

class SOM(basic.Basic):

	def __init__(self, *page, data, hidden_neurons=[100], initial_learning_rate=0.8, max_epoches=10, least_correct_rate=None, mode=1):
		super().__init__(*page)
		self.page.page_component.print_to_result("Init: SOM")
		self.data = data
		self.learning_rate = initial_learning_rate
		self._total_epoches = max_epoches
		self._least_correct_rate = least_correct_rate
		self.nn_structure = round(math.sqrt(hidden_neurons[0]))
		self.mode = mode

		self.current_iterations = 0
		self.stop_thread = False
		self._correct_rate = 0
		self._epoches_print_rate = max(1, self._total_epoches//10)
		self.neighbor_range = 2

		self._print_init_set()
		self._initial_neurons(self.nn_structure)

	''' Call by thread '''
	def run(self):
		self.start_training()
		

	def _initial_neurons(self, square_weight):
		dim = self.data.x.shape[1]
		network_architecture = []
		''' hidden layers '''
		for _ in range(square_weight):
			network_architecture.append((square_weight, dim))

		self.weights = ns.LayersImplement().create_neurons_layer(network_architecture)
		self.page.page_component.print_to_result("\n=== Init weights ===")
		self.page.page_component.print_to_result(self.weights)

	def start_training(self): 
		while self.current_iterations < self._total_epoches or (self.mode==2 and self._least_correct_rate > current_correct_rate):
			## User stop process
			if(self.stop_thread): 
				self.page.page_component.print_to_result("Interrupt training ...")
				return
			## Traing Step Alg
			self._train_step()
			self.current_iterations += 1
			self.data.shuffle_data()
			## print current Epoches state
			if(self.current_iterations%self._epoches_print_rate == 0):
				self.page.page_component.print_to_result("\n======Epoches {}======".format(self.current_iterations))
				_, current_correct_rate = self._cal_correct_rate("Training", self.data.train_x, self.data.train_y)
				_, current_correct_rate = self._cal_correct_rate("Testing", self.data.test_x, self.data.test_y)
				result, _ 				= self._cal_correct_rate("All", self.data.x, self.data.labels)
				self.page.graph.draw_result(self.data.x, result, self.weights, draw_weight=True)
	''' 
	'''
	def stop_training(self):
		self.stop_thread = True

	"""
	min_BMU = (0, 0, 10000), x1:i, x2:j, x3:dist
	"""
	def _train_step(self):
		for index in range(len(self.data.train_x)):
			min_BMU = self._find_BMU(self.data.train_x[index])
			self._adjust_weight(min_BMU, self.neighbor_range,self.data.train_x[index])

	""" best_matching_unit """
	def _find_BMU(self, intputX):
		intputX = intputX.T
		min_BMU = (0, 0, 10000)
		for i, weights in enumerate(self.weights):
			temp_minj, temp_mindist = self.__cal_Euclidean(intputX, weights)
			if( temp_mindist < min_BMU[2] ):
				min_BMU = (i, temp_minj, temp_mindist)
		# sys.exit(0)
		return min_BMU
	def __cal_Euclidean(self, V, Wi):
		Wi.result = np.linalg.norm(Wi.weight - V, axis=1)
		temp_minj = np.argmin(Wi.result)
		# print(V)
		# print(Wi)
		# print(Wi.result)
		# print(temp_minj)
		return temp_minj, Wi.result[temp_minj]

	def _adjust_weight(self, min_BMU, neighbor_range, intputX):
		square_range = (max(0, min_BMU[0]-neighbor_range), min(min_BMU[0]+neighbor_range, self.nn_structure-1) )
		for i in range(square_range[0], square_range[1]):
			for j in range(square_range[0], square_range[1]):
				self.weights[i].weight[j] = self.weights[i].weight[j] + self.learning_rate * (intputX - self.weights[i].weight[j])
	''' 
	'''

	def _cal_correct_rate(self, dataformat, datasetX, datasetY):
		super(SOM, self)._cal_correct_rate(dataformat, datasetX, datasetY)

		self.page.page_component.print_to_result("Data correct rate: {}/{}, {}%  \nError rate       : {}%  \n"\
													 .format(0, 0, 0, 0) )
		return datasetY, 1

	def _pass_activation_function(self, weight_output):
		""" pass neuron """
	@property
	def data(self):
		return self._data

	@data.setter
	def data(self, data):
		self.page.page_component.print_to_result("\n=== Load dataset ===")
		self._data = Data(data)

	def _print_init_set(self):
		super(SOM, self)._print_init_set()
		self.page.page_component.print_to_result("Learning rate: {}".format(self.learning_rate))
		self.page.page_component.print_to_result("Totale epoches: {}".format(self._total_epoches))
		self.page.page_component.print_to_result("Mode: {}".format(self.mode))
		self.page.page_component.print_to_result("Epoches print rate: {}".format(self._epoches_print_rate))
		self.page.page_component.print_to_result("NN structure size: {}".format(self.nn_structure))