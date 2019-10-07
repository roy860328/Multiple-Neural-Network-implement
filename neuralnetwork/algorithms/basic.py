import abc
from ..utils.utils import Data 

class Basic(abc.ABC):
	def __init__(self, data, total_epoches, initial_learning_rate):
		super().__init__()
		self.data = data
		self.current_iterations = 0
		self._total_epoches = total_epoches
		self._learning_rate = initial_learning_rate
		self._correct_rate = 0

	def run(self):
		for self.current_iterations in range(self._total_epoches):
			self.train_step()
	@abc.abstractmethod
	def train_step(self):
		""" each train step with the training algorithm """
	@abc.abstractmethod
	def cal_correct_rate(self):
		""" compare current output with data """

	@property
	def data(self):
		return self._data

	@data.setter
	def data(self, data):
		self._data = Data()
		self._data.load_data(data)