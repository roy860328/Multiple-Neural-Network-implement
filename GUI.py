import tkinter as tk
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import sys
#
import numpy as np
from Neural_Network import Neural_Network

class GUI():
    def __init__(self):
        self.setInterface()
    def setInterface(self):

        interface = tk.Tk()
        # 創造視窗x
        interface.title('interface')
        interface.geometry('800x800')
        # 學習率字幕
        learnrate = tk.Label(interface, text="learnrate")
        learnrate.pack()
        # input learnrate
        self.learnrateentry = tk.Entry(interface)
        self.learnrateentry.insert(0, "0.5")
        self.learnrateentry.pack()
        # 收斂字幕
        convergence = tk.Label(interface, text="convergence (train times)")
        convergence.pack()
        # input convergence
        self.convergenceentry = tk.Entry(interface)
        self.convergenceentry.insert(0, "10")
        self.convergenceentry.pack()
        # 列出txt檔案
        self.listTxt = tk.Listbox(interface)
        ##############os.path.dirname(sys.executable)當產出exe檔時才能正確找到txt檔案位置,但無法在.py檔中使用
        ##############os.getcwd()只有在.py檔有用,因為exe檔的默認位置在"cd ~" 讀檔時會找不到檔案
        print("sys.executable directory: ", os.path.dirname(sys.executable))
        # os.chdir(os.path.dirname(sys.executable))
        haveTxt = ''
        for file in os.listdir(os.getcwd() + "/data"):
            if file.endswith(".txt") or file.endswith(".TXT"):
                haveTxt += str(file) + ','
        haveTxt = haveTxt.split(",")
        haveTxt = list(filter(None, haveTxt))
        for txt in haveTxt:
            self.listTxt.insert(0, txt)
        # listTxt.bind('<<ListboxSelect>>', fileSelection)
        self.listTxt.pack()
        # 訓練按鈕
        trainbtn = tk.Button(interface, text="train", command=self.clickTrainBtn)
        trainbtn.pack()
        # outputresult
        outputresult = tk.Label(interface, text="outputresult")
        outputresult.pack()
        # input learnrate
        self.outputresultprint = tk.StringVar()
        outputresultLabel = tk.Label(interface, textvariable=self.outputresultprint)
        outputresultLabel.pack()
        #
        # 讓視窗實現
        interface.mainloop()

    def clickTrainBtn(self):
        ###clear 之前的plot
        plt.clf()
        ###偵測列表選取的txt檔案   #################################listTxt如何取得的？
        selectionfile = self.listTxt.curselection()
        selectionfile = self.listTxt.get(selectionfile)
        array = self.readFile(selectionfile)
        ###取得學習率跟收斂條件
        lrate = float(self.learnrateentry.get())
        ccondition = int(self.convergenceentry.get())

        trainrate, testrate, weight = Neural_Network().train(array, ccondition, lrate)
        self.showTrainresult(trainrate, testrate, weight)
        plt.show()

    def showTrainresult(self, trainrate, testrate, weight):
        printString = "\n" + "trainrate: " + str(trainrate) + "\n" + "testrate: " + str(testrate) + "\n"
        # [(printString += "weight[" + str(i) + "]: " + str(weight[i]) + "\n") for i in range(weight.shape[0])]
        for i in range(weight.shape[0]):
            printString += "weight[" + str(i) + "]: " + str(weight[i]) + "\n"
            self.outputresultprint.set(printString)
    def readFile(self, file):
        try:
            string = ""
            script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
            rel_path = "data/" + file
            abs_file_path = os.path.join(script_dir, rel_path)
            pfile1 = open(abs_file_path, "r")
            string = pfile1.read()
            string = string.split('\n')
            # string to double list
            string = [i.split(' ') for i in string]
            string = [x for x in string if x != ['']]
            string = np.array(string, dtype=float)
        except Exception as e:
            print(e)

        return string


