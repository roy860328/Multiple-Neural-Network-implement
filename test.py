__author__ = 'Dania'
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import *

colors = ['maroon', 'goldenrod', 'red', 'darkorange', 'peachpuff']


x=np.array ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
v= np.array ([16,16.31925,17.6394,16.003,17.2861,17.3131,19.1259,18.9694,22.0003,22.81226])
p= np.array ([16.23697,     17.31653,     17.22094,     17.68631,     17.73641 ,    18.6368,
	19.32125,     19.31756 ,    21.20247  ,   22.41444   ,  22.11718  ,   22.12453])

class mclass:
	def __init__(self,  window):
		self.window = window
		self.box = Entry(window)
		self.button = Button (window, text="check", command=self.plot2)
		self.box.pack ()
		self.button.pack()
		
		self.frame = Frame(self.window)
		self.frame.pack(side=LEFT, fill=BOTH, padx=1, pady=3, ipadx=2, ipady=5)

		self.frame = Frame(self.window)
		self.frame.pack(side=RIGHT, fill=BOTH, padx=1, pady=3, ipadx=2, ipady=5)
		
		self.frame = Frame(self.frame)
		self.frame.pack(fill=BOTH, padx=1, pady=3, ipadx=2, ipady=5)
		self.f = Figure(figsize=(3, 3), dpi=100)
		self.canvas = FigureCanvasTkAgg(self.f, master=self.frame)
		self.canvas.get_tk_widget().pack(side=RIGHT, fill=BOTH, expand=True, padx=2, pady=3)
		self.plot2()
		# self.plot2()
		# self.plot2()

	# def plot (self):

	# 	fig = Figure(figsize=(6,6))
	# 	a = fig.add_subplot(111)
	# 	a.scatter(v,x,color='red')
	# 	a.plot(p, range(2 +max(x)),color='blue')
	# 	a.invert_yaxis()

	# 	a.set_title ("Estimation Grid", fontsize=16)
	# 	a.set_ylabel("Y", fontsize=14)
	# 	a.set_xlabel("X", fontsize=14)

	# 	self.canvas = FigureCanvasTkAgg(fig, master=self.window)
	# 	self.canvas.get_tk_widget().pack()
	# 	self.canvas.draw()

	def plot2(self):
		global x
		self.canvas.draw()
		self.f.clf()
		a = self.f.add_subplot(111)
		colors = ['b', 'r', 'y', 'g']
		for i in range(len(x)):
			if int(x[i]) == 1: a.scatter(x[i], x[i], c=colors[0])
			else: a.scatter(x[i], x[i], c=colors[1])
		x = np.append(x, len(x)+1)
		self.canvas.draw()
		# self.window.update()

window= Tk()
start= mclass (window)
window.mainloop()