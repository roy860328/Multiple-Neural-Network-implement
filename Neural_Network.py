import numpy as np
import Render_Graph
import  sys

class Neural_Network():
    def __init__(self):
        print("init neural")
        self.convergenceCondition = 100
        self.learnrate = 0.5

    def train(self, array, convergenceCondition=100, learnrate=0.5):
        self.convergenceCondition = convergenceCondition
        self.learnrate = learnrate

        inputx, outputy, row, col = self.initializeDatatoInputandOutput(array)

        trainDatasIndex, trainDatas, testDatasIndex, testDatas \
            = self.chose_Train_Test_Data(inputx, row)

        ##神經元(perceptron)y, weight初始化
        y = list()
        result = self.new_Layer(1, 3)
        y.append(result)
        result = self.new_Layer(1, 1)
        y.append(result)

        weight = list()
        w = self.new_Layer(col, 3)
        weight.append(w)
        w = self.new_Layer(3, 1)
        weight.append(w)
        ##儲存最後一次outputy結果，用來畫出圖形
        trainOutputResult = np.zeros(trainDatas.shape[0])
        testOutputResult = np.zeros(testDatas.shape[0])
        # 正確“訓練辨識”數, 正確“測試辨識”數
        trainIdentifyCorrect, testIdentifyCorrect = 0, 0

        self.start_Train(y, weight, outputy, trainDatas, trainDatasIndex)
        ############ start train ##############
        # for n in range(ccondition):
        #     for i in range(trainDatas.shape[0]):
        #
        #         for j in range(y.shape[0]):
        #             y[j] = self.calNetwork(weight[j], trainDatas[i])
        #             ###adjust weight[j]
        #             weight[j] = self.adjustWeight(y[j], weight[j], outputy[trainDatasIndex[i]], lrate,
        #                                      trainDatas[i], j)
        #
        #         ###計算訓練辨識率
        #         if self.judgeYResult(y, outputy[trainDatasIndex[i]]):
        #             trainIdentifyCorrect = trainIdentifyCorrect + 1
        #             ##紀錄最後一次outputy結果
        #             if n == ccondition - 1:
        #                 trainOutputResult[i] = outputy[trainDatasIndex[i]]
        #         ##紀錄最後一次outputy結果
        #         elif n == ccondition - 1:
        #             trainOutputResult[i] = -1
        #
        # # print訓練辨識率
        # print("traincorrectrate: ", (trainIdentifyCorrect / trainDatas.shape[0]) / ccondition)
        #
        # ############ test rate ##############
        # for i in range(testDatas.shape[0]):
        #
        #     for j in range(y.shape[0]):
        #         y[j] = self.calNetwork(weight[j], testDatas[i])
        #
        #     ###計算測試辨識率
        #     if self.judgeYResult(y, outputy[testDatasIndex[i]]):
        #         testIdentifyCorrect = testIdentifyCorrect + 1
        #         ##紀錄最後一次outputy結果
        #         if n == ccondition - 1:
        #             testOutputResult[i] = outputy[testDatasIndex[i]]
        #     ##紀錄最後一次outputy結果
        #     elif n == ccondition - 1:
        #         testOutputResult[i] = -1
        #
        # # print測試辨識率 和 weight[]
        # print("testcorrectrate: ", testIdentifyCorrect / testDatas.shape[0])
        # [print("weight[", i, "]: ", weight[i]) for i in range(y.shape[0])]

        # 如果沒用到weight[0]，則設成0
        if int(np.amin(outputy)) != 0:
            weight[0] = [0]

        ##########畫出圖形##########
        Render_Graph.showGraph(trainDatas, trainOutputResult, testDatas, testOutputResult, y, weight)

        return ((trainIdentifyCorrect / trainDatas.shape[0]) / convergenceCondition), (testIdentifyCorrect / testDatas.shape[0]), weight


    # Initialize the text file to inputx and outputy array
    def initializeDatatoInputandOutput(self, array):
        row, col = array.shape
        ###set up inputx and outputy
        # split inputx and outputy
        array = np.hsplit(array, [col - 1])
        inputx = array[0]
        outputy = array[1]
        # add threshold to inputx
        threshold = np.zeros((row, 1)) - 1
        inputx = np.hstack((threshold, inputx))
        return inputx, outputy, row, col

    #chose train's data and test's data randomly
    def chose_Train_Test_Data(self, inputx, row):
        # 選擇2/3的隨機訓練data
        trainDatasIndex = np.random.choice(inputx.shape[0], size=int(row * 2 / 3) + 1, replace=False)
        trainDatas = inputx[trainDatasIndex, :]
        # 選擇1/3的隨機測試data
        testDatasIndex = np.arange(0, row)
        testDatasIndex = set(testDatasIndex) - set(trainDatasIndex)
        testDatasIndex = list(testDatasIndex)
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
                for i in range(Datas.shape[0]):
                    for j in range(len(weight)):
                        y[j] = self.calNetwork(weight[j], Datas[i])

                    ##Back Propagation
                    weight[-1], dk = self.adjustOutputWeight(weight[-1],
                                                  outputy, y[-1], y[-2])
                    for j in range(len(weight)-2, -1, -1):
                        if j-1 >= 0:
                            weight[j], dk = self.adjustHiddenWeight(weight[j],
                                                                    y[j-1], y[j], dk, weight[j+1])
                        #要修改第一層時，inout為Datas
                        else:
                            weight[j], dk = self.adjustHiddenWeight(weight[j],
                                                                Datas[i], y[j], dk, weight[j+1])
        except Exception as e:
            print(e)
            raise
        return y, weight
    def calNetwork(self, weight, datax):
        y = np.dot(weight, datax)
        ###sgn[y]
        y = np.exp(y)
        return y

    #Back Propagation
    def adjustOutputWeight(self, weight, expectoutputj, outputyj, outputyi):
        try:
            expectoutputj = (expectoutputj - np.amin(expectoutputj))/(np.amax(expectoutputj) - np.amin(expectoutputj))
            dk = (expectoutputj-outputyj)*outputyj*(1-expectoutputj)
            weight = weight + self.learnrate * dk.T * outputyi
        except Exception as e:
            print(e)
            raise
        return weight, dk

    #Back Propagation
    def adjustHiddenWeight(self, weightji, input, outputyj, dk, weightkj):
        dj = outputyj*(1-outputyj)*np.dot(dk, weightkj)
        weightji = weightji + self.learnrate * dj.T * input
        return weightji, dj

    # after calNetwork, if the result is correct  return True, else False
    def judgeYResult(self, y, yRealValue):
        if y[int(yRealValue)] == 1:
            nonzero = np.nonzero(y)[0]
            if nonzero.shape[0] == 1:
                return True
        return False



