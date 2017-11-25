import numpy as np
import Render_Graph
import  sys

class Neural_Network():
    def __init__(self):
        print("init neural")
        self.convergenceCondition = 100
        self.learnrate = 0.5
        #normalize in
        self.interval = 0
        self.min0 = False

    def train(self, array, convergenceCondition=100, learnrate=0.3):
        self.convergenceCondition = convergenceCondition
        self.learnrate = learnrate

        inputx, orioutput, row, col = self.initializeDatatoInputandOutput(array)
        outputy = orioutput
        outputy = self.normalizeExpectOutput(outputy)

        trainDatasIndex, trainDatas, testDatasIndex, testDatas \
            = self.chose_Train_Test_Data(inputx, row)

        ##神經元(perceptron)y, weight初始化
        NeuronNumber = 5

        y = list()
        result = self.new_Layer(1, NeuronNumber)
        y.append(result)
        result = self.new_Layer(1, 1)
        y.append(result)

        weight = list()
        w = self.new_Layer(col, NeuronNumber)
        weight.append(w)
        w = self.new_Layer(NeuronNumber+1, 1)
        weight.append(w)

        y, weight = self.start_Train(y, weight, outputy, trainDatas, trainDatasIndex)
        trainCorrectRate, trainOutputResult = self.cal_Multiple_Neural_Network_CorrectRate(y, weight, outputy, trainDatasIndex, trainDatas)
        testCorrectRate, testOutputResult = self.cal_Multiple_Neural_Network_CorrectRate(y, weight, outputy, testDatasIndex, testDatas)

        Render_Graph.showGraph(trainDatas, trainOutputResult, testDatas, testOutputResult, orioutput, self.interval)

        return trainCorrectRate, testCorrectRate


    # Initialize the text file to inputx and outputy array
    def initializeDatatoInputandOutput(self, array):
        row, col = array.shape
        ###set up inputx and outputy
        # split inputx and outputy
        array = np.hsplit(array, [col - 1])
        inputx = array[0]
        outputy = array[1]
        # add threshold to inputx
        # threshold = np.zeros((row, 1)) - 1
        # inputx = np.hstack((threshold, inputx))
        return inputx, outputy, row, col

    #Normalize expectoutput
    def normalizeExpectOutput(self, expectoutput):
        try:
            if np.amin(expectoutput) == 0:
                self.interval = np.amax(expectoutput) - np.amin(expectoutput)
                expectoutput = (expectoutput - np.amin(expectoutput)) / (self.interval)
                self.interval = self.interval + 1
                self.min0 = True
            else:
                self.interval = np.amax(expectoutput) - np.amin(expectoutput) + 1
                expectoutput = (expectoutput - np.amin(expectoutput)) / (self.interval)
        except Exception as e:
            print(e)
            raise
        return expectoutput


    #chose train's data and test's data randomly
    def chose_Train_Test_Data(self, inputx, row):
        # 選擇2/3的隨機訓練data
        if inputx.shape[0] > 4:
            trainDatasIndex = np.random.choice(inputx.shape[0], size=int(row * 2 / 3) + 1, replace=False)
            trainDatas = inputx[trainDatasIndex, :]
            # 選擇1/3的隨機測試data
            testDatasIndex = np.arange(0, row)
            testDatasIndex = set(testDatasIndex) - set(trainDatasIndex)
            testDatasIndex = list(testDatasIndex)
            testDatas = inputx[testDatasIndex, :]
        else:
            trainDatasIndex = np.arange(0, row)
            trainDatas = inputx[trainDatasIndex, :]
            # 選擇1/3的隨機測試data
            testDatasIndex = np.arange(0, row)
            testDatas = inputx[testDatasIndex, :]
        return trainDatasIndex, trainDatas, testDatasIndex, testDatas

    #weight = [outputSize, inputSize]  (outputSize = output Neuron's number)
    def new_Layer(self, inputSize, outputSize):
        weight = np.zeros(shape=(outputSize, inputSize))
        for i in range(outputSize):
            weight[i] = np.random.rand(1, inputSize)
        return weight

    # calculate network and adjust result to two value (0 or 1)
    def start_Train(self, y, weight, outputy, Datas, DatasIndex):
        try:
            outputy = outputy[DatasIndex,:]
            for _ in range(self.convergenceCondition):
                #store old weight to continue Back Propagation
                for i in range(Datas.shape[0]):
                    tempw = weight[:]
                    y[0] = self.calNetwork(weight[0], Datas[i])
                    for j in range(1, len(weight)):
                        y[j] = self.calNetwork(weight[j], y[j-1])

                    ##Back Propagation
                    weight[-1], dk = self.adjustOutputWeight(tempw[-1],
                                                             outputy[i], y[-1], y[-2])
                    for j in range(len(weight)-2, -1, -1):
                        if j-1 >= 0:
                            weight[j], dk = self.adjustHiddenWeight(tempw[j],
                                                                    y[j-1], y[j], dk, tempw[j+1])
                        #要修改第一層時，inout為Datas
                        else:
                            weight[j], dk = self.adjustHiddenWeight(tempw[j],
                                                                    Datas[i], y[j], dk, tempw[j+1])
        except Exception as e:
            print(e)
            raise
        return y, weight
    def calNetwork(self, weight, datax):
        datax = np.insert(datax, 0, -1)
        y = np.dot(weight, datax)
        ###sgn[y]
        y = 1/(1 + np.exp(-y))
        return y

    #Back Propagation
    def adjustOutputWeight(self, weight, expectoutputj, outputyj, outputyi):
        try:
            outputyi = np.insert(outputyi, 0, -1)
            dk = (expectoutputj-outputyj)*outputyj*(1-outputyj)
            weight = weight + self.learnrate * dk.T * outputyi
        except Exception as e:
            print(e)
            raise
        return weight, dk

    #Back Propagation
    def adjustHiddenWeight(self, weightji, input, outputyj, dk, weightkj):
        try:
            input = np.insert(input, 0, -1)
            input = np.array([input])
            weightkj = np.delete(weightkj, 0)
            dj = outputyj*(1-outputyj)*(dk * weightkj)
            dj = np.array([dj])
            dj = dj.T
            weightji = weightji + self.learnrate * np.dot(dj, input)
        except Exception as e:
            print(e)
            raise
        return weightji, dj

    def cal_Multiple_Neural_Network_CorrectRate(self, y, weight, outputy, DatasIndex, Datas):
        try:
            correctRate = 0
            outputy = outputy[DatasIndex, :]
            calOutputy = np.array(outputy, copy=True)
            for i in range(outputy.shape[0]):
                y[0] = self.calNetwork(weight[0], Datas[i])
                for j in range(1, len(weight)):
                    y[j] = self.calNetwork(weight[j], y[j - 1])

                #Lower and Upper Bound
                y[-1], calOutputy[i] = self.findBound(y)
                if self.judgeYResult(y[-1], outputy[i]):
                    correctRate += 1
            correctRate = correctRate/Datas.shape[0]
        except Exception as e:
            print(e)
            raise
        return correctRate, calOutputy
    #Lower and Upper Bound
    def findBound(self, y):
        intervalnumber = 1/(self.interval)
        tempty = y[-1]
        for j in range(int(self.interval)):
            if intervalnumber * j <= tempty and tempty <= intervalnumber * (j+1):
                tempty = intervalnumber * (j)
                if self.min0 == True:
                    tempty = tempty*2
                tempty = np.array([tempty])
                return tempty, tempty

    # after calNetwork, if the result is correct  return True, else False
    def judgeYResult(self, outputy, expectoutput):
        if outputy[0] == expectoutput[0]:
            return True
        return False



