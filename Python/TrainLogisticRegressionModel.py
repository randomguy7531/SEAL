import collections
import LogisticRegressionModel
import EvaluationsStub

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
    if('0' in input):
        return 0
    else:
        return 1

path = "LogisticRegressionTrainData.csv"
f = open(path, 'r')
lines = f.readlines()
x = []
y = []
for l in lines:
    splitl = l.split(sep=',')
    x.append([parseFeatureString(i) for i in splitl[0:9]])
    y.append(parseFeatureString(splitl[10]))

(xTrain, yTrain, xTest, yTest) = TrainTestSplit(x, y)

model = LogisticRegressionModel.LogisticRegressionModel(10)
print("Logistic regression model")
trainingLosses = []
testLosses = []
testAccuracies = []
numIterationsThousands = 7
for i in range(numIterationsThousands):
    print("iteration {0}".format(i*1000))

    yTrainPredicted = model.predict(xTrain)
    yTestPredicted = model.predict(xTest)

    trainLoss = model.loss(xTrain, yTrain)
    testLoss = model.loss(xTest, yTest)

    trainAccuracy = EvaluationsStub.Accuracy(yTrain,yTrainPredicted)
    testAccuracy = EvaluationsStub.Accuracy(yTest, yTestPredicted)

    trainingLosses.append([i*1000,trainLoss])

    if((i % 10) == 0):
        testLosses.append([i*1000,testLoss])
        testAccuracies.append([i*1000,testAccuracy])
    
    model.fit(xTrain, yTrain, iterations=(1000), step=0.01)

yTrainPredicted = model.predict(xTrain)
yTestPredicted = model.predict(xTest)

trainLoss = model.loss(xTrain, yTrain)
testLoss = model.loss(xTest, yTest)

trainAccuracy = EvaluationsStub.Accuracy(yTrain,yTrainPredicted)
testAccuracy = EvaluationsStub.Accuracy(yTest, yTestPredicted)

trainingLosses.append([numIterationsThousands*1000,trainLoss])
testLosses.append([numIterationsThousands*1000,testLoss])
testAccuracies.append([numIterationsThousands*1000,testAccuracy])

print(model.weights)

with open("trainingLosses_logisticRegression.csv", "w+") as file:
    for line in trainingLosses:
        file.write("{0},{1}\n".format(line[0], line[1]))

with open("testLosses_logisticRegression.csv", "w+") as file:
    for line in testLosses:
        file.write("{0},{1}\n".format(line[0], line[1]))

with open("testAccuracies_logisticRegression.csv", "w+") as file:
    for line in testAccuracies:
        file.write("{0},{1}\n".format(line[0], line[1]))

with open("testWeights_logisticRegression.csv", "w+") as file:
    for weight in model.weights:
        file.write("{0},".format(weight))
    file.write("\n")


EvaluationsStub.ExecuteAll(yTest, yTestPredicted)

