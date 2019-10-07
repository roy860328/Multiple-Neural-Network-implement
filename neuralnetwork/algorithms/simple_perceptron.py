from . import Basic

class SimplePerceptron(Basic):

	def __init__(self, data, neurons=2, total_epoches=10, initial_learning_rate=0.8):
		super.__init__(data, total_epoches, initial_learning_rate)
		self.neurons = neurons
		self._initial_neurons()

	def _initial_neurons(self):
		pass
	def train_step(self):
		pass
	def _forward_propagation(self):
		pass
	def _adjust_weight(self):
		pass
	def cal_correct_rate(self):
		pass