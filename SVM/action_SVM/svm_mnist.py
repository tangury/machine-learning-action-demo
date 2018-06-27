from numpy import *
from time import sleep

def loadDataSet(fileName):        # 加载数据，矩阵化
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

def selectJrand(i,m): # 随机参数函数设置
    j = i
    while(j == i):
        j = int(random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):   # 给定阿尔法的范围
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def calcEk(oS, k):
        fxk = float(multiply(oS.alphas, oS.labelMat).T * oS.k[:,k] + oS.b) #预测值
        Ek = fxk - float(oS.labelMat[k]) #　误差缓存值
        return Ek

def selectJ(i, oS, Ei): #　内循环启发方式
        maxK = -1; maxDeltaE = 0; Ej = 0
        oS.eCache[i] = [1, Ei] #　误差缓存的范围
        validEcacheList = nonzero(oS.eCache[:,0].A)[0] # 返回误差值不为０的下标，放到有效误差缓存列表中
        if(len(validEcacheList)) > 1:
            for k in validEcacheList:
                if k == i: continue
                Ek = calcEk(oS, k) #　计算第ｋ个向量的误差
                deltaE = abs(Ei - Ek) # 第ｉ个向量和第ｋ个相量差的绝对值
                if(deltaE > maxDeltaE): # 取最大的误差差，并求相应相量的下标
                    maxK = k; maxDeltaE = deltaE; Ej = Ek
            return maxK, Ej
        else:
            j = selectJrand(i, oS.m) # 否则随机去下标ｊ
            Ej = calcEk(oS, j) # 计算ｊ对应的误差
        return j, Ej

def updateEk(oS, k): #　更新误差
        Ek = calcEk(oS, k)
        oS.eCache[k] = [1, Ek]

def innerL(i, oS):
    Ei = calcEk(oS, i)
    if((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or \
            ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        j, Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();
        if(oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H: print("L == H"); return 0
        eta = 2.0 * oS.k[i,j] - oS.k[i, i] - oS.k[j,j]
        if eta >= 0: print("eta >= 0"); return 0
        oS.alphas[j] -= oS.labelMat[j] *(Ei -Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)
        if(abs(oS.alphas[j] -alphaJold) < 0.00001):
            print("j not moving enough"); return 0
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
        updateEk(oS, i)
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.k[i,i]- \
            oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.k[i,j]
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.k[i,j] - \
            oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.k[j,j]
        if(0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
        elif(0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
        else: oS.b = (b1 + b2) / 2.0
        return  1
    else: return 0

def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup = ('lin', 0)):
    oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler, kTup)
    iter = 0
    entireSet = True; alphaPairsChanged = 0
    while(iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
                print("fullSet, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        else:
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
                iter += 1
        if entireSet: entireSet = False
        elif(alphaPairsChanged == 0): entireSet = True
        print("iteration number: %d" % iter)
    return oS.b, oS.alphas

def calcWs(alphas, dataArr, classLabels):
    X = mat(dataArr); labelMat = mat(classLabels).transpose()
    m, n = shape(X)
    w = zeros((n, 1))
    for i in range(m):
        w += multiply(alphas[i] * labelMat[i], X[i,:].T)
    return w
def kernelTrans(X, A, kTup):
    m, n = shape(X)
    k = mat(zeros((m,1)))
    if kTup[0] == 'lin':k = X * A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j,:] - A
            k[j] = deltaRow * deltaRow.T
        k = exp(k / (-1 * kTup[1] ** 2))
    else: raise NameError('Houston We Have a Problem -- That Kernel is not recognized')
    return k


class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.eCache = mat(zeros((self.m, 2)))
        self.k = mat(zeros((self.m, self.m)))
        for i in range(self.m):
            self.k[:,] = kernelTrans(self.X, self.X[i,:], kTup)

def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def loadImages(dirName):
    from os import listdir
    hwLabels = []
    trainingFileList = listdir(dirName)
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9: hwLabels.append(-1)
        else:hwLabels.append(1)
        trainingMat[i,:] = img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat, hwLabels


def testDigits(kTup = ('rbf', 10)):
    dataArr, labelArr = loadImages('trainingDigits')
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    dataMat = mat(dataArr); labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A > 0)[0]
    sVs = dataMat[svInd]
    labelSV = labelMat[svInd]
    print("there are %d Support Vectors" % shape(sVs)[0])
    m, n = shape(dataMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i,:], kTup)
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]): errorCount += 1
    print("the training error rate is: %f" % (float(errorCount) / m))
    dataArr, labelArr = loadImages('testDigits')
    errorCount = 0
    dataMat = mat(dataArr); labelMat = mat(labelArr).transpose()
    m, n = shape(dataMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i,:], kTup)
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]): errorCount += 1
    print("the test error rate is: %f" % (float(errorCount) / m))

if "__name__ == __main__":
    testDigits()