import csv
import pandas as pd
import codecs
from binary_perceptron import Perceptron
from logistic_regression import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if __name__ == '__main__':

    raw_data = pd.read_csv('train_binary.csv', header=0)
    data = raw_data.values

    imgs = data[0::, 1::]
    labels = data[::, 0]

    test_time = 10

    p = Perceptron()
    lr = LogisticRegression()

    # writer = writer_csv_file('result.csv')
    # with open('result.csv', 'wb', encoding='utf-8') as f:
    #     writer = csv.writer(f,)

    csvfile = codecs.open('result.csv', 'w+', encoding='utf-8')
    writer = csv.writer(csvfile)

    for time in range(test_time):
        print ('iterater time %d' % time)

        train_features, test_features, train_labels, test_labels = train_test_split(
            imgs, labels, test_size=0.33, random_state=23323)

        p.train(train_features, train_labels)
        lr.train(train_features, train_labels)

        p_predict = p.predict(test_features)
        lr_predict = lr.predict(test_features)

        p_score = accuracy_score(test_labels, p_predict)
        lr_score = accuracy_score(test_labels, lr_predict)

        print ('perceptron accruacy score ', p_score)
        print ('logistic Regression accruacy score ', lr_score)



        writer.writerow([time,p_score,lr_score])
