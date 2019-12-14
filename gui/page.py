import abc
import tkinter as tk
from tkinter import ttk
import numpy as np
import threading
import queue

from .guicomponent import PageComponent, Graph
from neuralnetwork.algorithms import simpleperceptron, mlp, som

''' GUI function implement (Called by GUI) '''
class Pages(abc.ABC):
	def __init__(self, root):
		self.root = tk.Frame(root)
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
	def __init__(self, root, dataset_list):
		super().__init__(root)
		self.create_para_IO_frame(dataset_list)
		self.create_graph_frame()
		self.root.pack(side=tk.LEFT, fill=tk.BOTH)
		
		# self.start_to_train()

	def create_para_IO_frame(self, dataset_list):
		self.IO_frame = tk.Frame(master=self.root)
		self.page_component = PageComponent(self.IO_frame, dataset_list)
		
		self.page_component.data_select()
		self.page_component.learning_rate()
		self.page_component.convergence_condition()
		self.page_component.execution_button(self.start_to_train, self.stop_to_start)
		self.page_component.training_result()
		
		self.IO_frame.pack(side=tk.LEFT, fill=tk.BOTH)

	def create_graph_frame(self):
		self.graph_frame = tk.Frame(master=self.root)
		self.graph = Graph(self.graph_frame)
		self.graph_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=2, pady=3)

		# self.page_component.graph_None()
		# self.page_component.graph_2D(self.root)
		# self.page_component.graph_3D()
	def start_to_train(self):
		self.page_component.print_to_result("\n\n\n=== Start to Train ===")

		self.page_component.start_to_train()
		kwargs = dict(data=self.page_component.dataset_list[self.page_component.data_selection.get()],
					  initial_learning_rate=float(self.page_component.learning_rate.get()),
					  max_epoches=int(self.page_component.max_epoches.get()),
					  least_correct_rate=float(self.page_component.least_correct_rate.get()),
					  )
		if (self.page_component.is_condition_max_epoches.get()==2):
			kwargs["mode"] = "least_correct_rate"
		self.neural_network = simpleperceptron.SimplePerceptron(self, **kwargs)
		self.neural_network.run()

	def stop_to_start(self):
		if(self.neural_network != None):
			self.page_component.stop_to_start()
			self.neural_network.stop_training()

	def finish_training(self):
		self.page_component.finish_training()


class MLPPages(Pages):
	def __init__(self, root, dataset_list):
		super().__init__(root)
		self.create_para_IO_frame(dataset_list)
		self.create_graph_frame()
		self.root.pack(side=tk.LEFT, fill=tk.BOTH)
		
		# self.start_to_train()

	def create_para_IO_frame(self, dataset_list):
		self.IO_frame = tk.Frame(master=self.root)
		self.page_component = PageComponent(self.IO_frame, dataset_list)
		
		self.page_component.data_select()
		self.page_component.learning_rate()
		self.page_component.convergence_condition()
		self.page_component.neurons_layers()
		self.page_component.execution_button(self.start_to_train, self.stop_to_start)
		self.page_component.training_result()
		
		self.IO_frame.pack(side=tk.LEFT, fill=tk.BOTH)

	def create_graph_frame(self):
		self.graph_frame = tk.Frame(master=self.root)
		self.graph = Graph(self.graph_frame)
		self.graph_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=2, pady=3)

	def start_to_train(self):
		self.page_component.print_to_result("\n\n\n=== Start to Train ===")

		self.page_component.start_to_train()
		kwargs = dict(data=self.page_component.dataset_list[self.page_component.data_selection.get()],
					hidden_neurons=list(map(int, self.page_component.hidden_layer.get().split(" "))),
					initial_learning_rate=float(self.page_component.learning_rate.get()),
					max_epoches=int(self.page_component.max_epoches.get()),
					least_correct_rate=float(self.page_component.least_correct_rate.get()),
					)
		if (self.page_component.is_condition_max_epoches.get()==2):
			kwargs["mode"] = "least_correct_rate"
		self.neural_network = mlp.MLP(self, **kwargs)
		self.neural_network.run()

	def stop_to_start(self):
		if(self.neural_network != None):
			self.page_component.stop_to_start()
			self.neural_network.stop_training()

	def finish_training(self):
		self.page_component.finish_training()



class SOMPages(Pages):
	def __init__(self, root, dataset_list):
		super().__init__(root)
		self.create_para_IO_frame(dataset_list)
		self.create_graph_frame()
		self.root.pack(side=tk.LEFT, fill=tk.BOTH)

		self.request_queue = queue.Queue()
		self.result_queue = queue.Queue()
		self.main_thread_draw_update()

		# self.start_to_train()

	def create_para_IO_frame(self, dataset_list):
		self.IO_frame = tk.Frame(master=self.root)
		self.page_component = PageComponent(self.IO_frame, dataset_list)
		
		self.page_component.data_select()
		self.page_component.learning_rate(0.1)
		self.page_component.convergence_condition(1000)
		self.page_component.neurons_layers(100)
		self.page_component.execution_button(self.start_to_train, self.stop_to_start)
		self.page_component.training_result()
		
		self.IO_frame.pack(side=tk.LEFT, fill=tk.BOTH)

	def create_graph_frame(self):
		self.graph_frame = tk.Frame(master=self.root)
		self.graph = Graph(self.graph_frame)
		self.graph_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=2, pady=3)

	def start_to_train(self):
		self.page_component.print_to_result("\n\n\n=== Start to Train ===")

		if len(list(map(int, self.page_component.hidden_layer.get().split(" ")))) != 1:
			self.page_component.hidden_layer.set(64)

		self.page_component.start_to_train()
		kwargs = dict(data=self.page_component.dataset_list[self.page_component.data_selection.get()],
					hidden_neurons=list(map(int, self.page_component.hidden_layer.get().split(" "))),
					initial_learning_rate=float(self.page_component.learning_rate.get()),
					max_epoches=int(self.page_component.max_epoches.get()),
					least_correct_rate=float(self.page_component.least_correct_rate.get()),
					)
		if (self.page_component.is_condition_max_epoches.get()==2):
			kwargs["mode"] = 2
		self.neural_network = som.SOM(self, **kwargs)

		thread_run_neural_network = threading.Thread(target=self.neural_network.run)
		thread_run_neural_network.daemon = True
		thread_run_neural_network.start()

	def stop_to_start(self):
		if(self.neural_network != None):
			self.page_component.stop_to_start()
			self.neural_network.stop_training()
			self.clear_queue()
	def finish_training(self):
		self.page_component.finish_training()
		self.stop_to_start()

	def main_thread_draw_update(self):
		try:
			callable, args, kwargs = self.request_queue.get_nowait()
		except queue.Empty:
			pass
		else:
			# print("Run === self.graph.draw_result ===\n\n\n")
			retval = callable(*args, **kwargs)
			self.result_queue.put(retval)
		self.root.after(10, self.main_thread_draw_update)
	
	def clear_queue(self):
		self.request_queue = queue.Queue()
		self.result_queue = queue.Queue()

	def set_thread_draw_update(self, *args, **kwargs):
		self.request_queue.put((self.graph.draw_result, args, kwargs))
