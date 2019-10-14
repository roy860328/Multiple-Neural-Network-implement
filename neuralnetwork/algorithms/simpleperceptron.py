from . import basic
from neuralnetwork.neurons import neurons as ns
import threading 


class SimplePerceptron(basic.Basic):

	def __init__(self, *page, data, hidden_neurons=[1], initial_learning_rate=0.8, max_epoches=10, least_error_rate=None, mode="max_epoches"):
		hidden_neurons = [1]
		super().__init__(data, hidden_neurons, initial_learning_rate, max_epoches, least_error_rate, mode, *page)
		self.page = page[0]
		self.page.page_component.print_to_result("Init: SimplePerceptron")

	''' Call by thread '''
	def run(self):
		super(SimplePerceptron, self).run()
		
	def start_training(self): 
		super(SimplePerceptron, self).start_training()
		try: 
			pass
		finally: 
			pass

	def _adjust_weight(self, intputX, outputY):	
		# print(self.weights[0].result)
		# print(outputY)
		if self.weights[0].result[0] == 0 and outputY == 1:
			# print("train")
			self.weights[0].weight = self.weights[0].weight + self.learning_rate * intputX
		elif self.weights[0].result[0] == 1 and outputY == 0:
			# print("train2")
			self.weights[0].weight = self.weights[0].weight - self.learning_rate * intputX

	def _pass_activation_function(self, weight_output):
		return list(map(lambda label: 1 if label > 0 else 0, weight_output.flatten()))
