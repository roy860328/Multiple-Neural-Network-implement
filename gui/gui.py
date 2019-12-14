import abc
import tkinter as tk
from tkinter import ttk
import numpy as np

from .page import SimplePerceptronPages, MLPPages, SOMPages


class GUI():
	def __init__(self, *args):
		self.interface = tk.Tk()
		self.interface.title("Neural Network GUI")
		self.interface.geometry("1000x500")
		self.interface.resizable(False, False)
		
		self.base_frame = tk.Frame(master=self.interface)
		self.base_frame.pack()
		self._init_windows(args)
	def _init_windows(self, args):

		main_notebook  = ttk.Notebook(self.base_frame)
		main_notebook.pack(fill=tk.BOTH, padx=2, pady=3)		

		page1 = tk.Frame(main_notebook)
		SimplePerceptronPages(page1, args[0])
		main_notebook.add(page1, text="SimplePerceptron")

		page2 = tk.Frame(main_notebook)
		MLPPages(page2, args[0])
		main_notebook.add(page2, text="MultilayerPerceptron")

		page3 = tk.Frame(main_notebook)
		SOMPages(page3, args[0])
		main_notebook.add(page3, text="SOM")

		main_notebook.select(page3)

	def run_GUI(self):
		self.interface.mainloop()

