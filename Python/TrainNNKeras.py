import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from keras.layers import Activation
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.models import Sequential
from keras.layers import Dense
import numpy

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

def square_activation(x):
    return x * x

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

#get_custom_objects().update({'custom_activation': Activation(custom_activation)})

dataset = numpy.loadtxt("NeuralNetBinaryClassTrainTestData.txt", delimiter="\t")
X = dataset[:,0:10]
Y = dataset[:,10]

(xTrain, yTrain, xTest, yTest) = TrainTestSplit(X, Y)

model = Sequential()
model.add(Dense(12, input_dim=10, init='uniform', activation=square_activation))
model.add(Dense(8, init='uniform', activation=square_activation))
model.add(Dense(1, init='uniform', activation='sigmoid'))

# compile model
model.compile(loss='binary_crossentropy' , optimizer='adam', metrics=['accuracy'])

# input the dataset into created model
model.fit(xTrain, yTrain, nb_epoch=150, batch_size=10)

# evaluate the model on the training set
scores = model.evaluate(xTrain, yTrain)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# evaluate the model on the test set
scores = model.evaluate(xTest, yTest)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

predictions = model.predict(xTest)
print(predictions)

print(model.get_weights())

print(model.input)
print(model.layers[0].output)