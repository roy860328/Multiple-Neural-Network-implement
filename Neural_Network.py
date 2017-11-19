import numpy as np
import Render_Graph

class Neural_Network():
    def __init__(self):
        print("init neural")

    def train(self, array, ccondition=100, lrate=0.5):
        inputx, outputy, row, col = self.initializeDatatoInputandOutput(array)

        trainDatasIndex, trainDatas, testDatasIndex, testDatas \
            = self.chose_Train_Test_Data(inputx, row)

        ##神經元(perceptron)outputy數量 weight初始化
        y = self.new_Layer(1, int(np.amax(outputy)) + 1)
        weight = self.new_Layer(col, y.shape[0])
        ##儲存最後一次outputy結果，用來畫出圖形
        trainOutputResult = np.zeros(trainDatas.shape[0])
        testOutputResult = np.zeros(testDatas.shape[0])
        # 正確“訓練辨識”數, 正確“測試辨識”數
        trainIdentifyCorrect, testIdentifyCorrect = 0, 0

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

        return ((trainIdentifyCorrect / trainDatas.shape[0]) / ccondition), (testIdentifyCorrect / testDatas.shape[0]), weight


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
    def start_Train(self, runtimes ):
        return runtimes
    def calNetwork(self, weight, datax):
        y = np.dot(weight, datax)
        ###sgn[y]
        if y > 0:
            y = 1
        else:
            y = 0
        return y
    # y=計算的結果, weight=當前權重, outputy=正確輸出, lrate=學習率, trainDatas=當前inputx, expectoutput=期望訓練的值
    def adjustWeight(self, y, weight, outputy, lrate, trainDatas, expectoutput):
        if y == 0 and outputy == expectoutput:
            weight = weight + np.multiply(lrate, trainDatas)
        elif y == 1 and outputy != expectoutput:
            weight = weight - np.multiply(lrate, trainDatas)
        return weight


    # after calNetwork, if the result is correct  return True, else False
    def judgeYResult(self, y, yRealValue):
        if y[int(yRealValue)] == 1:
            nonzero = np.nonzero(y)[0]
            if nonzero.shape[0] == 1:
                return True
        return False



