import tkinter as tk
from tkinter import ttk
import numpy as np
import threading 
from pprint import pprint

import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

colors = ['maroon', 'goldenrod', 'red', 'darkorange', 'peachpuff', 'k']


class PageComponent(object):
	"""docstring for PageComponent"""
	def __init__(self, root, dataset_list):
		super(PageComponent, self).__init__()
		self.root = root
		self.dataset_list = dataset_list
	'''

	'''
	def data_select(self):
		panel = tk.LabelFrame(self.root, text="Select dataset: ")
		data_value = tk.StringVar()
		keepvalue = data_value.get()
		self.data_selection = ttk.Combobox(panel, textvariable=keepvalue, values=list(self.dataset_list.keys()), state='readonly')
		self.data_selection.current(0)
		# self.data_selection.bind("<<ComboboxSelected>>", lambda x: self.graph.check_dataset_diemension())
		self.data_selection.pack(fill=tk.X)
		panel.pack(padx=2, pady=3, fill=tk.BOTH)
	'''

	'''
	def learning_rate(self):
		panel = tk.LabelFrame(self.root, text="Learning Rate: ")
		self.learning_rate = tk.Entry(panel)
		self.learning_rate.config(width=5)
		self.learning_rate.insert(tk.END, 0.05)
		self.learning_rate.pack(side=tk.LEFT)
		panel.pack(fill=tk.BOTH, padx=1, pady=3, ipadx=2, ipady=5)

	''' 
	convergence condition 
	rellated function: convergence_condition_max_epoches, convergence_condition_min_correct_rate, switch_convergence_condition
	'''
	def convergence_condition(self):
		panel = tk.LabelFrame(self.root, text="Convergence Condition: max_epoches, min_correct_rate")
		self.is_condition_max_epoches = tk.IntVar()
		self.is_condition_max_epoches.set(1)
		self.convergence_condition_max_epoches(panel)
		self.convergence_condition_min_correct_rate(panel)
		panel.pack(fill=tk.BOTH, padx=1, pady=3, ipadx=2, ipady=5)

	def convergence_condition_max_epoches(self, panel):
		self.set_ler_checkbtn = tk.Checkbutton(panel, text="Max Epoches", variable=self.is_condition_max_epoches, command=self.switch_convergence_condition, onvalue = 1, offvalue = 2)
		self.set_ler_checkbtn.pack(side=tk.LEFT)
		self.max_epoches = tk.Entry(panel)
		self.max_epoches.config(width=7)
		self.max_epoches.insert(tk.END, 5000)
		self.max_epoches.pack(side=tk.LEFT)
	def convergence_condition_min_correct_rate(self, panel):
		self.set_ler_checkbtn = tk.Checkbutton(panel, text="Least Correct Rate: ", variable=self.is_condition_max_epoches, command=self.switch_convergence_condition, onvalue = 2, offvalue = 1)
		self.set_ler_checkbtn.pack(side=tk.LEFT)
		self.least_correct_rate = tk.Entry(panel)
		self.least_correct_rate.insert(tk.END, 80)
		self.least_correct_rate.config(width=3, state='disable')
		self.least_correct_rate.pack(side=tk.LEFT)
		tk.Label(panel, text="%").pack(side=tk.LEFT)

	def switch_convergence_condition(self):
		if self.is_condition_max_epoches.get() == 1:
			self.max_epoches.config(state='normal')
			self.least_correct_rate.config(state='disable')
		else:
			self.max_epoches.config(state='disable')
			self.least_correct_rate.config(state='normal')

	'''

	'''
	def neurons_layers(self, nn_structure=(5, 5, 5)):
		panel = tk.LabelFrame(self.root, text="Neurons Hidden Layers: ")
		self.hidden_layer = tk.Entry(panel)
		self.hidden_layer.config(width=25)
		self.hidden_layer.insert(tk.END, nn_structure)
		self.hidden_layer.pack(side=tk.LEFT)
		panel.pack(fill=tk.BOTH, padx=1, pady=3, ipadx=2, ipady=5)
	'''

	'''
	def execution_button(self, start_to_train, stop_to_start):
		panel = tk.LabelFrame(self.root, text="Execution: ")
		self.execute = tk.Button(panel, text='Train', width=10, height=2, padx=5, command=start_to_train)
		self.execute.pack(side=tk.LEFT)
		self.stop_button(panel, stop_to_start)

	def stop_button(self, panel, stop_to_start):
		self.stop = tk.Button(panel, text='Stop', width=10, height=2, padx=5, command=stop_to_start)
		self.stop.pack(side=tk.LEFT)
		panel.pack(fill=tk.BOTH, padx=1, pady=3, ipadx=2, ipady=5)

	def start_to_train(self):
		self.execute.config(state='disable')
		self.data_selection.config(state='disable')
		
		self.stop.config(state='normal')

	def stop_to_start(self):
		self.execute.config(state='normal')
		self.data_selection.config(state='readonly')
		
		self.stop.config(state='disable')

	def finish_training(self):
		self.execute.config(state='normal')
		self.data_selection.config(state='readonly')

	'''

	'''
	def training_result(self):
		panel = tk.LabelFrame(self.root, text="Training Result: ")
		self.console = tk.Text(panel, height=20, width=50, state='disable')
		console_sb = tk.Scrollbar(panel)
		console_sb.config(command=self.console.yview)
		self.console.config(yscrollcommand=console_sb.set)
		self.console.pack(side=tk.LEFT)
		console_sb.pack(side=tk.LEFT, fill=tk.Y)
		panel.pack(padx=2, pady=3, fill=tk.BOTH)
	'''

	'''
	def print_to_result(self, input_obj):
		if isinstance(input_obj, str):
			pass
		elif isinstance(input_obj, list):
			input_obj = str(list(map(str, input_obj)))
			# if isinstance(input, np.ndarray):
			# 	pass
		elif isinstance(input_obj, np.ndarray):
			input_obj = np.array2string(input_obj)

		print(input_obj)
		# pprint(input_obj)
		self.console.config(state='normal')
		self.console.insert(tk.END, input_obj + '\n')
		self.console.config(state='disable')
		self.console.see(tk.END)


class Graph():
	"""docstring for Graph"""
	def __init__(self, root):
		self.root = root
		# self.component = page.page_component

		self.creat_2D_canvas()
		# self.creat_3D_canvas()
		# self.create_blank_canvas()
		# self.root.pack(fill=tk.BOTH)
		
		# self.show_2D_canvas()

	''' Create Canvas '''
	def creat_2D_canvas(self):
		self.graphic_2d_frame = tk.Frame(self.root)
		self.graphic_2d_figure = Figure(figsize=(6, 6), dpi=100)
		self.canvas_2d = FigureCanvasTkAgg(self.graphic_2d_figure, self.graphic_2d_frame)
		self.canvas_2d.get_tk_widget().pack(fill=tk.BOTH, expand=False, padx=2, pady=3)

	def creat_3D_canvas(self):
		self.graphic_3d_page = tk.Frame(self.root)
		figure = Figure(figsize=(6, 6), dpi=100)
		self.canvas_3d = FigureCanvasTkAgg(figure, self.graphic_3d_page)
		self.canvas_3d.get_tk_widget().pack(expand=True)
		self.graphic_3d = figure.add_subplot(1, 1, 1, projection=Axes3D.name)
		self.surface_list = [None] * 4 
		self.graphic_3d_zlim = None

	def create_blank_canvas(self):
		self.no_graphic_page = tk.Frame(self.root)
		tk.Label(self.no_graphic_page, text="").pack(expand=True)

	''' Canvas controll '''
	def draw_result(self, inputX=None, result=None, weights=None, draw_weight=False):
		assert inputX.shape[0] == result.shape[0], "Error: inputX&Y size not same"

		dim = inputX.shape[1]
		canvas_type = ""
		if dim == 2:
			canvas_type = "Show 2D canvas"
			self._show_2D_canvas(inputX, result, weights, draw_weight)
		elif dim == 3:
			canvas_type = "Show 3D canvas"
			self._show_3D_canvas(inputX, result)
		else:
			canvas_type = "Unshow canvas"
			self._unshow_canvas()


	def _show_2D_canvas(self, inputX=None, result=None, weights=None, draw_weight=False):
		# self.graphic_3d_page.pack_forget()
		# self.no_graphic_page.pack_forget()
		self.graphic_2d_frame.pack(expand=True)

		self.graphic_2d_figure.clf()
		fig = self.graphic_2d_figure.add_subplot(1, 1, 1)
		self.__draw_2D_point(fig, inputX, result)
		if(len(weights)==1): self.__draw_2D_line(fig, weights)
		if(draw_weight): self.__draw_weight_mesh(fig, weights)

		self.canvas_2d.draw()

	def __draw_2D_point(self, fig, inputX=None, result=None):
		[fig.scatter(float(x[0]),float(x[1]), c=colors[int(result[index])]) for index, x in enumerate(inputX)]
		fig.set_title ("Estimation Grid", fontsize=16)
		fig.set_ylabel("X2", fontsize=14)
		fig.set_xlabel("X1", fontsize=14)

	def __draw_2D_line(self, fig=None, weights=None):
		weight = weights[0]()
		x = np.arange(-2, 2, 0.1)
		for i in range(weight.shape[0]):
			slope = -(weight[i][0]/weight[i][1])
			intercept = weight[i][2]/weight[i][1]
			print(slope, intercept)
			fig.plot(x, slope*x + intercept)

	def __draw_weight_mesh(self, fig=None, weights=None):
		for i in range(weights[0].weight.shape[0]):
			for j in range(len(weights)):
				w = weights[i].weight[j]
				# print(w)
				if(i < weights[0].weight.shape[0]-1):
					w1 = weights[i+1].weight[j]
					fig.plot([w[0], w1[0]], [w[1], w1[1]], linestyle='-', c='black', linewidth=0.5, marker='o')
				if(j < weights[0].weight.shape[1]-1):
					w2 = weights[i].weight[j+1]
					fig.plot([w[0], w2[0]], [w[1], w2[1]], linestyle='-', c='black', linewidth=0.5, marker='o')
		# print(weights)
		# print(len(weights))
		# print(weights[0])
	def _show_3D_canvas(self, result):
		self.graphic_2d_frame.pack_forget()
		self.no_graphic_page.pack_forget()
		self.graphic_3d_page.pack(expand=True)
		
	def _unshow_canvas(self):
		self.graphic_2d_frame.pack_forget()
		self.graphic_3d_page.pack_forget()
		self.no_graphic_page.pack(expand=True)