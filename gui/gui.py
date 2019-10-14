import abc
import tkinter as tk
from tkinter import ttk
import numpy as np

from .guicomponent import PageComponent, Graph
from neuralnetwork.algorithms import simpleperceptron


class GUI():
	def __init__(self, *args):
		self.interface = tk.Tk()
		self.interface.title("Neural Network GUI")
		self.interface.geometry("1000x500")
		self.interface.resizable(False, False)

		self.base_frame = tk.Frame(master=self.interface).pack()
		self._init_windows(args)
	def _init_windows(self, args):
		main_notebook  = ttk.Notebook(self.base_frame)
		main_notebook.pack(fill=tk.BOTH, padx=2, pady=3)
		
		p1 = SimplePerceptronPages(args[0])
		main_notebook.add(p1.root, text="First tab")

		# f1 = tk.Frame(main_notebook)
		# #Add the tab
		# main_notebook.add(f1, text="Two tab")

	def run_GUI(self):
		self.interface.mainloop()


''' GUI function implement (Called by GUI) '''
class Pages(abc.ABC):
	def __init__(self):
		self.root = tk.Frame()
		self.neural_network = None

	@abc.abstractmethod
	def create_para_IO_frame(self):
		""" """
	@abc.abstractmethod
	def create_graph_frame(self):
		""" """
	@abc.abstractmethod
	def start_to_train(self):
		""" """
	@abc.abstractmethod
	def stop_to_start(self):
		""" """
	@abc.abstractmethod
	def finish_training(self):
		""" """

class SimplePerceptronPages(Pages):
	def __init__(self, dataset_list):
		super().__init__()
		self.page_component = PageComponent(self.root, dataset_list)
		self.create_para_IO_frame()
		self.graph = Graph(self)
		self.create_graph_frame()
		self.root.pack()
		
		self.start_to_train()

	def create_para_IO_frame(self):
		self.page_component.data_select()
		self.page_component.learning_rate()
		self.page_component.convergence_condition()
		self.page_component.neurons_layers()
		self.page_component.execution_button(self.start_to_train, self.stop_to_start)
		self.page_component.training_result()
		
		self.page_component.root.pack(side=tk.LEFT, fill=tk.BOTH)
		

	def create_graph_frame(self):
		# self.graph.
		# self.graph.root.pack(side=tk.RIGHT, fill=tk.BOTH)
		pass

	def start_to_train(self):
		self.page_component.start_to_train()
		kwargs = dict(data=self.page_component.dataset_list[self.page_component.data_selection.get()],
					hidden_neurons=list(map(int, self.page_component.hidden_layer.get().split(" "))),
					initial_learning_rate=float(self.page_component.learning_rate.get()),
					max_epoches=int(self.page_component.max_epoches.get()),
					least_error_rate=float(self.page_component.least_error_rate.get()),
					)
		# print(kwargs["data"])
		if (self.page_component.is_condition_max_epoches.get()==2):
			kwargs["mode"] = "least_error_rate"
		self.neural_network = simpleperceptron.SimplePerceptron(self, **kwargs)
		self.neural_network.start()

	def stop_to_start(self):
		if(self.neural_network != None):
			self.page_component.stop_to_start()
			self.neural_network.stop_training()

	def finish_training(self):
		self.page_component.finish_training()
		self.graph.draw_result()