from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(fileName):      # 加载数据文件
    numFeat = len(open(fileName).readline().split('\t')) # 剔除符号等，获取数据行数
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)   # 获得训练集
        labelMat.append(float(curLine[-1])) # 获得标签集
    return dataMat,labelMat


def stumpClassify(dataMatrix, dimen, threshVal, threshIneq): #　通过阈值对数据进行分类函数
    retArray = ones((shape(dataMatrix)[0], 1))
    if threshIneq == 'lt': # 在大于和小于之间切换
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray

def buildStump(dataArr, classLabels,D):
    dataMatrix = mat(dataArr); labelMat = mat(classLabels).T
    m,n = shape(dataMatrix) # 输入数据集的大小
    numSteps = 10.0; bestStump = {}; bestclasEst = mat(zeros((m, 1))) # 初始化
    minError = inf # 初始化inf为无穷大
    for i in range(n): # 在数据集上所有特征进行遍历
        rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max();  # 取数据集中的最大值和最小值
        stepSize = (rangeMax - rangeMin) / numSteps   # 遍历范围
        for j in range(-1, int(numSteps) + 1): # 在(-1,11)之间遍历
            for inequal in ['lt', 'gt']: # 在大于和小于之间切换不等式
                threshVal = (rangeMin + float(j) * stepSize)  # 确定阈值大小
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal) #　分类结果
                errArr = mat(ones((m, 1)))  # 将分类错误的置为１，对的为０
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T * errArr # 将错误向量与权重向量相乘并求和，　计算加权错误率
                print("split: dim %d, thresh %.2f, thresh inequal: %s, the weighted error id %.3f" %  \
                      (i, threshVal, inequal, weightedError))   # 打印出相应的衡量指标值
                if weightedError < minError: # 找到误差率最小的值并将其保存，返回
                    minError = weightedError
                    bestclasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestclasEst


def adaBoostTrainDs(dataArr, classLabels, numIt = 40): # 训练更新Ｄ向量
    weakClassArr = []
    m = shape(dataArr)[0] #　输入数据行数
    D = mat(ones((m, 1)) / m) # 初始化Ｄ值
    aggClassEst = mat(zeros((m, 1))) # 初始化每个数据点类别估计值
    for i in range(numIt): # 迭代次数
        bestStump, error, classEst = buildStump(dataArr, classLabels, D) # 计算出最佳单层决策树，错误率，分类结果向量
        print("D:", D.T) # 打印更新的Ｄ向量
        alpha = float(0.5 * log((1.0 - error) / max(error, 1e-16))) # 根据公式计算阿尔法的值,1e-16避免除零溢出
        bestStump['alpha'] = alpha # 将阿尔法的值保存
        weakClassArr.append(bestStump)  # 将最佳单层决策树的分类结果等加入列表
        print("classEst:", classEst.T) # 打印预测分类结果向量
        expon = multiply(-1 * alpha * mat(classLabels).T, classEst) # 将公式程序化，计算下一次Ｄ
        D = multiply(D, exp(expon))
        D = D / D.sum()
        aggClassEst += alpha * classEst # 给单层最佳决策树分类器赋权重，并相加
        print("aggClassEst: ",aggClassEst.T)
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))
        errorRate = aggErrors.sum() / m #　错误率累加计算平均错误率
        print("total error: ", errorRate, "\n") # 打印错误率
        if errorRate == 0.0: break # 若错误率为０提前结束迭代
    return weakClassArr, aggClassEst

def adaClassify(datToClass, classifierArr): # 利用训练出的多个弱分类器函数
    dataMatrix = mat(datToClass) # 将数据转换为numpy
    m = shape(dataMatrix)[0]   # 读取待分类样列的个数
    aggClassEst = mat(zeros((m, 1))) # 构建０列向量
    for i in range(len(classifierArr)): # 遍历所有若分类器
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'],\
                                 classifierArr[i]['thresh'], classifierArr[i]['ineq']) # 分类测试集数据
        aggClassEst += classifierArr[i]['alpha'] * classEst # 各分类器的结果向量乘以相应的权重得出总模型分类结果
        print(aggClassEst)
    return sign(aggClassEst) # 分类结果大于０则返回＋１，小于０则返回－１


def plotRoc(preStrengths, classLabels): # 第一个参数为预测强度，第二个参数为分类标签
    cur = (1.0, 1.0)
    ySum = 0.0 # 初始化相关参数
    numPosClas = sum(array(classLabels) == 1.0) # 计算正列的数目
    yStep = 1 / float(numPosClas)
    xStep = 1 / float(len(classLabels) - numPosClas) #  确定x,y轴的步长
    sortedIndicies = preStrengths.argsort() #　按照从小到大的顺序排序绘制
    fig = plt.figure()  # 初始化画笔
    fig.clf()
    ax = plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0; delY = yStep;
        else:
            delX = xStep; delY = 0;
            ySum += cur[1]
        ax.plot([cur[0], cur[0] - delX], [cur[1], cur[1] - delY], c = 'b') # 每得到一个标签为1.0的类，　则要沿
    # y轴的方向下降一个步长，即不断降低假阳率．对于每个其他标签，则是在x轴方向上倒退一步长（假阴率方向）
        cur = (cur[0] - delX, cur[1] - delY)
    ax.plot([0, 1], [0, 1],'b--')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('Roc curve for AdaBoost Horse Colic Detection System')
    ax.axis([0,1,0,1])
    plt.show()
    print("the Area Under the Curve is: ", ySum * xStep)

if "__name__ == __main__":
    datArr, labelArr = loadDataSet('horseColicTraining2.txt')
    classifierArray, aggClassEst = adaBoostTrainDs(datArr, labelArr, 10)
    plotRoc(aggClassEst.T, labelArr)

