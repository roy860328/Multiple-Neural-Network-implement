from gui.gui import GUI
import os
from os.path import isfile, join

def main():
	data = read_file("data/PerceptDataSet/basic")
	data.update(read_file("data/PerceptDataSet/advanced"))
	gui = GUI(data)
	# gui.interface.mainloop()
	gui.run_GUI()

def read_file(files_path):
	files_name = os.listdir(files_path)
	files_path = [os.path.join(files_path, name) for name in files_name if isfile(join(files_path, name))]
	print(files_path)
	dataset_list = {}
	for idx, dataset in enumerate(list(map(lambda x: open(x, 'r'),files_path))):
		data = []
		for line in dataset:
			data.append(list(map(float, line.split(" "))))
		# print(data)
		dataset_list[files_name[idx]] = data
	# print(dataset_list)

	# dataset_list = {'ff': 'asd', 'aaa': 'sad'}
	return dataset_list

if __name__ == "__main__":
	main()