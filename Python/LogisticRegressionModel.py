import math
class LogisticRegressionModel(object):
    """A logistic regression model"""

    def __init__(self, numFeaturesWithoutBias):
        # init weights (note presence of weight for bias/constant term)
        self.weights = [0.0] * (numFeaturesWithoutBias + 1)
        pass

    def add_bias_term(self, x):
        x_with_bias = []
        for x_i in x:
            x_with_bias.append( x_i + [1])
        return x_with_bias

    def fit(self, x, y, iterations, step):
        x_with_bias = self.add_bias_term(x)
        N = len(x_with_bias)
            
        for iteration in range(iterations):
            partials = [0 for i in range(len(self.weights))]
            y_hat = self.score(x_with_bias)
            for i in range(N):
                x_i = x_with_bias[i]
                y_i = y[i]
                y_hat_i = y_hat[i]
                for j in range(len(x_i)):
                    partials[j] += ((y_hat_i - y_i) * x_i[j]) / N
            for i in range(len(self.weights)):
                self.weights[i] -= (step * partials[i])

    def score(self, x):
        predictions = []

        for example in x:
            z = sum([ example[i] * self.weights[i] for i in range(len(example)) ])
            predictions.append(1.0 / (1.0 + math.exp(-1.0 * z)))
        
        return predictions

    def score_independent(self, x):
        x_with_bias = self.add_bias_term(x)
        predictions = []

        for example in x_with_bias:
            z = sum([ example[i] * self.weights[i] for i in range(len(example)) ])
            predictions.append(1.0 / (1.0 + math.exp(-1.0 * z)))
        
        return predictions

    def predict(self, x):
        x_with_bias = self.add_bias_term(x)
        scores = self.score(x_with_bias)
        predictions = [ ]

        for score in scores:
            if score >= 0.5:
                predictions.append(1)
            else:
                predictions.append(0)

        return predictions

    def loss(self, x, y):
        loss = 0
        x_with_bias = self.add_bias_term(x)
        y_hat = self.score(x_with_bias)
        for i in range(len(x)):
            y_hat_i = y_hat[i]
            y_i = y[i]
            if(y_i == 1):
                loss += -1.0 * math.log(y_hat_i)
            elif(y_i == 0):
                loss += -1.0 * math.log(1 - y_hat_i)
        return loss