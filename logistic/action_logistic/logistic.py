from numpy import *
import matplotlib.pyplot as plt


def loadDataSet():            #加载数据
    dataMat = []; labelMat = []
    fr = open('testSet.txt')     #打开textSet.txt文本
    for line in fr.readlines():
        lineArr = line.strip().split()   #去除首尾空格
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])   #将处理好的训练集数据转换为浮点型加到数组列表中
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

def sigmoid(inX):        #定义sigmoid激活函数
    return 1.0 / (1 + exp(-inX))


def gradeAscent(dataMatIn, classLabels):      #定义梯度上升函数
    dataMatrix = mat(dataMatIn)               #将训练集的数据转换成矩阵
    labelMat = mat(classLabels).transpose()   #将测试集的数据转换为矩阵并转置
    m, n = shape(dataMatrix)                  #读取训练集数据的大小
    alpha = 0.001
    maxCycles = 500
    weights = ones((n, 1))                    #初始化权重
    for k in range(maxCycles):                #训练迭代次数
        h = sigmoid(dataMatrix * weights)     #预测值
        error = (labelMat - h)                #误差
        weights = weights + alpha *dataMatrix.transpose() * error   #权重更新公式，利用了链式求导
    return weights


def plotBestFit(weights):          #绘制决策边界图
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]         #读取训练集的行数
    xcord1 = []; ycord1 = []      #建立标签分类集
    xcord2 = []; ycord2 = []
    for i in range(n):            #分类标签
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1]); ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1]); ycord2.append(dataArr[i, 2])
    fig = plt.figure()                   #绘图设置
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s = 30, c = 'red', marker = 's')
    ax.scatter(xcord2, ycord2, s = 30, c = 'green')
    x = arange(-3.0, 3.0, 0.1)            #定义输入x的取值(注：Python3.x中range的浮点型应为arange)
    y = (-weights[0] - weights[1] * x) / weights[2]   #定义目标直线：w0x0+w1*x1+w2x2
    ax.plot(x, y)
    plt.xlabel('x1')     #设置x轴为x1
    plt.ylabel('x2')     #设置y轴为y1
    plt.show()

def stocGradAscent0(dataMatrix, classLabels):     #随机梯度上升函数定义
    m, n = shape(dataMatrix)       #读取训练集数据
    alpha = 0.01                   #学习率
    weights = ones(n)               #初始化权重
    for i in range(m):              #更新权重
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights


def stocGradAscent1(dataMatrix, classLabels, numIter = 150):    #随机梯度提升算法
    m, n = shape(dataMatrix)          # 读取训练集数据行列
    weights = ones(n)
    for j in range(numIter):           # 迭代次数最大150
        dataIndex = list(range(m))     # 训练数据样本个数
        for i in range(m):
            alpha = 4 / (1.0 + j + i)+0.0001    # 更新学习率
            randIndex = int(random.uniform(0, len(dataIndex)))   # 特征索引值
            h = sigmoid(sum(dataMatrix[randIndex] * weights))    # 将输入通入预测函数
            error = classLabels[randIndex] - h                   # 计算误差
            weights = weights + alpha * error * dataMatrix[randIndex]  # 更新权重
            del(dataIndex[randIndex])
    return weights

def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5 : return 1.0
    else: return 0.0

def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currline = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currline[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currline[21]))
    trainWeights = stocGradAscent1(array(trainingSet),trainingLabels, 500)
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currline = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currline[i]))
        if int(classifyVector(array(lineArr), trainWeights)) != int(currline[21]):
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec)
    print("the error rate of this test is : %f" % errorRate)
    return errorRate


def multiTest():
    numTests = 10; errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is: \
          %f" % (numTests, errorSum / float(numTests)))

if __name__ == '__main__':
    multiTest()
    dataArr, labelMat = loadDataSet()
    weights = gradeAscent(dataArr, labelMat)
    plotBestFit(weights.getA())







