import numpy as np
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB

def TrainTestSplit(x, y, percentTest = .25):
    if(len(x) != len(y)):
        raise UserWarning("Attempting to split into training and testing set.\n\tArrays do not have the same size. Check your work and try again.")

    numTest = round(len(x) * percentTest)

    if(numTest == 0 or numTest > len(y)):
        raise UserWarning("Attempting to split into training and testing set.\n\tSome problem with the percentTest or data set size. Check your work and try again.")

    xTest = x[:numTest]
    xTrain = x[numTest:]
    yTest = y[:numTest]
    yTrain = y[numTest:]

    return (xTrain, yTrain, xTest, yTest)

def parseFeatureString(input):
    return float(input)

path = "NaiveBayesTrainTestData.txt"
f = open(path, 'r')
lines = f.readlines()
x = []
y = []
for l in lines:
    splitl = l.split(sep='\t')
    x.append([parseFeatureString(i) for i in splitl[0:10]])
    y.append(parseFeatureString(splitl[10]))

(xTrain, yTrain, xTest, yTest) = TrainTestSplit(x, y)
gnb = GaussianNB()
gnb.fit(xTrain,yTrain)
print(gnb.class_prior_)

testVal =[[1,0,1,0,0,0,0,0,0,0]]
print(gnb.predict(testVal))