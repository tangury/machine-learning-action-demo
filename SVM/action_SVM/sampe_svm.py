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


def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = mat(dataMatIn); labelMat = mat(classLabels).transpose() # 数据矩阵化
    b = 0; m, n = shape(dataMatrix) #　初始化相应参数
    alphas = mat(zeros((m, 1)))
    iter = 0
    while(iter < maxIter): #　设置迭代次数
        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i,:].T)) + b  # fXi为预测值，其矩阵为：n*i
            Ei = fXi - float(labelMat[i])    # 误差
            if((labelMat[i] * Ei < - toler) and (alphas[i] < C)) or ((labelMat[i] * Ei > toler) and (alphas[i] > 0)):   # 设定阿尔法的范围，及确定SVM间隔
                j =selectJrand(i, m) # 选择第二个阿尔法参数
                fXj = float(multiply(alphas, labelMat).T *(dataMatrix * dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j]) # 方法同第一个阿尔法
                alphaIold = alphas[i].copy();
                alphaJold = alphas[j].copy(); #　为两个阿尔法分配内存空间
                if(labelMat[i] != labelMat[j]): # 确定阿尔法ｊ的取值范围
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H: print("L == H"); continue #　若相等则继续
                eta = 2.0 * dataMatrix[i,:] * dataMatrix[j,:].T - dataMatrix[i,:] * dataMatrix[i,:].T \
                      - dataMatrix[j, :] * dataMatrix[j, :].T #　最优修改量
                if eta >= 0: print("eta >= 0"); continue
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta #　计算阿尔法ｊ的值
                alphas[j] = clipAlpha(alphas[j], H, L) #　设置阿尔法ｊ的取值范围在［Ｌ，Ｈ］
                if(abs(alphas[j] - alphaJold) < 0.00001): print("j not moving enough");continue
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j]) #　更新阿尔法ｉ
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i,:] * dataMatrix[i,:].T- \
                    labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[i,:] * dataMatrix[j,:].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i,:] * dataMatrix[j,:].T - \
                    labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[j,:] * dataMatrix[j,:].T # 计算两向量对应的ｂ值
                if(0 < alphas[i]) and (C > alphas[i]): b = b1
                elif(0 < alphas[j]) and (C > alphas[j]): b = b2
                else:b = (b1 + b2) / 2.0 # 根据阿尔法的大小确定b的取值
                alphaPairsChanged += 1 #　记录阿尔法的改变次数
                print("iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))

            if(alphaPairsChanged == 0): iter += 1
            else: iter = 0
            print("iteration number: %d" % iter)
        return b, alphas


if __name__ ==  '__main__':
    dataArr, labelArr = loadDataSet('testSet.txt')
    b,alphas = smoSimple(dataArr, labelArr, 0.6, 0.001, 1000)
    print(b)




