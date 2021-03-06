import numpy as np

class Neuron():
	"""docstring for Neuron"""
	def __init__(self, dim):
		self.weight = np.random.rand(1, dim)
		self.y = 0

''' Layers type'''
class NeuronsLayer():
	"""docstring for Neuron"""
	def __init__(self, shape, random_range=None):
		if(random_range):
			self.weight = np.random.uniform(low=random_range[0], high=random_range[1], size=(shape[0], shape[1]))
		else:
			self.weight = np.random.rand(shape[0], shape[1])
		self.result = np.zeros((shape[0], 1))
		self.deltas = np.zeros((shape[0], 1))
	def __call__(self):
		return self.weight
	def __str__(self):
		return np.array2string(self.weight)

		
class LayersImplement(object):
	"""docstring for LayersImplement"""
	def __init__(self):
		super(LayersImplement, self).__init__()
	'''  '''
	def create_neurons_layer(self, hidden_neurons, random_range=None):

		return [NeuronsLayer(shape, random_range) for shape in hidden_neurons]
		