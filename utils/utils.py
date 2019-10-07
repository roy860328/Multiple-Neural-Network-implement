import numpy as np

class Data():
	def __init__(self):
		self.load = None
		self.labels = None
		self.train_x = None
		self.train_y = None
		self.test_x = None
		self.test_y = None

	def load_data(self, data):
		self.load = data