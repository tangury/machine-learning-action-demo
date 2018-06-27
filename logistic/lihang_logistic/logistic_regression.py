import time
import math
import random

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class LogisticRegression(object):

    def __init__(self):               # 初始化学习率及最大迭代次数
        self.learning_step = 0.00001
        self.max_iteration = 5000

    def predict_(self,x):              # 分类器函数的选定与设置
        wx = sum([self.w[j] * x[j] for j in range(len(self.w))])
        exp_wx = math.exp(wx)

        predict1 = exp_wx / (1 + exp_wx)
        predict0 = 1 / (1 + exp_wx)

        if predict1 > predict0:
            return 1
        else:
            return 0


    def train(self,features, labels):
        self.w = [0.0] * (len(features[0]) + 1)     # 初始化权重

        correct_count = 0      # 初始化正确分类的个数
        time = 0               # 初始化时间

        while time < self.max_iteration:
            index = random.randint(0, len(labels) - 1)     # 生成标签索引
            x = list(features[index])                      # 标签索引对应的特征， 添加特征
            x.append(1.0)
            y = labels[index]

            if y == self.predict_(x):                      # 标签值与预测值相等？
                correct_count += 1
                if correct_count > self.max_iteration:      # 分类正确数大于最大迭代数，迭代结束
                    break
                continue

            # print 'iterater times %d' % time
            time += 1                                     # 迭代计数
            correct_count = 0

            wx = sum([self.w[i] * x[i] for i in range(len(self.w))])  # 求积和权重与输入
            exp_wx = math.exp(wx)

            for i in range(len(self.w)):
                self.w[i] -= self.learning_step * \
                    (-y * x[i] + float(x[i] * exp_wx) / float(1 + exp_wx))       # 更新权重


    def predict(self,features):
        labels = []

        for feature in features:
            x = list(feature)          # 获得训练输入特征
            x.append(1)                # 扩充x的存储
            labels.append(self.predict_(x))  # 获得预测值

        return labels

if __name__ == "__main__":
    print ('Start read data')

    time_1 = time.time()     # 训练开始时候的时间

    raw_data = pd.read_csv('train_binary.csv',header=0)    # 读取训练数据
    data = raw_data.values

    imgs = data[0::,1::]          # 分离训练的标签数据与特征数据
    labels = data[::,0]


    # 选取 2/3 数据作为训练集， 1/3 数据作为测试集
    train_features, test_features, train_labels, test_labels = train_test_split(imgs, labels, test_size=0.33, random_state=23323)

    time_2 = time.time()     # 读取数据时候的时间
    print ('read data cost ',time_2 - time_1,' second','\n')

    print ('Start training')
    lr = LogisticRegression()
    lr.train(train_features, train_labels)

    time_3 = time.time()          # 训练结束的时间
    print ('training cost ',time_3 - time_2,' second','\n')

    print ('Start predicting')     # 预测开始时候的时间
    test_predict = lr.predict(test_features)
    time_4 = time.time()
    print ('predicting cost ',time_4 - time_3,' second','\n')

    score = accuracy_score(test_labels,test_predict)
    print ("The accruacy socre is ", score)
