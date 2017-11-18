import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def showGraph(trainDatas, trainOutputResult, testDatas, testOutputResult, y, weight):
    try:
        plt.figure(1, figsize=(16, 12))
        plt.subplot(221)
        plt.title("TrainSample")
        showPlot(trainDatas, trainOutputResult, y.shape[0], weight)
        plt.subplot(222)
        plt.title("TestSample (black point is identify error data)")
        showPlot(testDatas, testOutputResult, y.shape[0], weight)
    except Exception as e:
        pass
# 可以plot出2D的data. outputResult為資料集輸出的結果(output).
def showPlot(Datas, outputResult, outputHadvalue, weight):
    Datas = np.hsplit(Datas, [1])
    Datas = np.hsplit(Datas[1], [1])

    # plt.scatter(Datas[0], Datas[1], c='r', label='perceptron1')
    pointlabel = np.zeros(outputHadvalue)
    colorSelect = ['black', 'b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i in range(Datas[0].shape[0]):
        plt.scatter(Datas[0][i], Datas[1][i], c=colorSelect[int(outputResult[i]) + 1],
                    label=str(int(outputResult[i])) if pointlabel[int(outputResult[i])] == 0 else "")
        if pointlabel[int(outputResult[i])] == 0:
            pointlabel[int(outputResult[i])] = 1

    plt.xlim([-5, 5])
    plt.ylim([-5, 8])

    x = np.arange(-5, 5, 0.1)
    for i in range(weight.shape[0]):
        y = -(weight[i][1] / weight[i][2]) * x - weight[i][0] / weight[i][2]
        plt.plot(x, y)
    plt.legend()
