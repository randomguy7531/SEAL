import numpy as np
from sklearn.linear_model import LinearRegression

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

path = "LinearRegressionTrainTestData.txt"
f = open(path, 'r')
lines = f.readlines()
x = []
y = []
for l in lines:
    splitl = l.split(sep='\t')
    x.append([parseFeatureString(i) for i in splitl[0:10]])
    y.append(parseFeatureString(splitl[10]))

(xTrain, yTrain, xTest, yTest) = TrainTestSplit(x, y)

reg = LinearRegression().fit(xTrain, yTrain)

print("coefficients and intercept...")
print(reg.coef_)
print(reg.intercept_)

yTestHat = reg.predict(xTest)

print(yTestHat)

xGenTest = [[43.45089541,53.15091973,53.07708932,93.65456934,65.23330105,69.34856259,62.91649012,35.28814156,108.1002775,100.1735266],
[51.59952075,99.48561775,95.75948428,126.6533636,142.5235433,90.97955769,43.66586306,85.31957886,62.57644682,66.12458533],
[94.77026243,71.51229208,85.33271407,69.58347566,107.8693045,101.6701889,89.88200921,54.93440139,105.5448532,72.07947083],
[89.53820766,100.199631,86.19911875,85.88717675,33.92249944,80.47113937,65.34411148,89.70004394,75.00778202,122.3514331],
[96.86101454,97.54597612,122.9960987,86.1281547,115.5539807,107.888993,65.51660154,74.17007885,48.04727402,93.56952259],
[91.75121904,121.2115065,62.92763365,99.4343452,70.420912,88.0580948,71.82993308,80.49171244,87.11321454,100.1459868]]

yGenTest = reg.predict(xGenTest)

print(yGenTest)