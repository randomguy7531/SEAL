

def __CheckEvaluationInput(y, yPredicted):
    # Check sizes
    if(len(y) != len(yPredicted)):
        raise UserWarning("Attempting to evaluate between the true labels and predictions.\n   Arrays contained different numbers of samples. Check your work and try again.")

    # Check values
    valueError = False
    for value in y:
        if value not in [0, 1]:
            valueError = True
    for value in yPredicted:
        if value not in [0, 1]:
            valueError = True

    if valueError:
        raise UserWarning("Attempting to evaluate between the true labels and predictions.\n   Arrays contained unexpected value. Must be 0 or 1.")

def Accuracy(y, yPredicted):
    __CheckEvaluationInput(y, yPredicted)

    correct = []
    for i in range(len(y)):
        if(y[i] == yPredicted[i]):
            correct.append(1)
        else:
            correct.append(0)

    return sum(correct)/len(correct)

def numCorrect(y, yPredicted):
    __CheckEvaluationInput(y, yPredicted)

    correct = []
    for i in range(len(y)):
        if(y[i] == yPredicted[i]):
            correct.append(1)
        else:
            correct.append(0)

    return sum(correct)

def Precision(y, yPredicted):
    falsePositives = 0
    truePositives = 0
    
    for i in range(len(y)):
        if(yPredicted[i] == 1):
            if(y[i] == 1):
                truePositives += 1
            else:
                falsePositives += 1

    #handle the no-positive-predictions case
    if((truePositives + falsePositives) == 0):
        return 1.0
        
    return truePositives / (truePositives + falsePositives)

def Recall(y, yPredicted):
    falseNegatives = 0
    truePositives = 0
    
    for i in range(len(y)):
        if(y[i] == 1):
            if(yPredicted[i] == 1):
                truePositives += 1
            else:
                falseNegatives += 1

    #handle the no-positive-labels case
    if((truePositives + falseNegatives) == 0):
        return 1.0

    return truePositives / (truePositives + falseNegatives)

def FalseNegativeRate(y, yPredicted):
    falseNegatives = 0
    negativePredictions = 0
    
    for i in range(len(y)):
            if(yPredicted[i] == 0):
                negativePredictions += 1
                if(y[i] == 1):
                    falseNegatives += 1

    #handle the no-negative-predictions case
    if(negativePredictions== 0):
        return 1.0

    return falseNegatives / negativePredictions

def FalsePositiveRate(y, yPredicted):
    falsePositives = 0
    negativeLabels = 0
    
    for i in range(len(y)):
            if(y[i] == 0):
                negativeLabels += 1
                if(yPredicted[i] == 1):
                    falsePositives += 1

    #handle the no-negative-labels case
    if(negativeLabels == 0):
        return 1.0

    return falsePositives / negativeLabels

def ConfusionMatrix(y, yPredicted):
    buckets = [0,0,0,0]
    for i in range(len(y)):
        # 2 bit encoding of result to bucket
        bucket = 2 * y[i] + yPredicted[i]
        buckets[bucket] += 1

    print("\t\t|\tLabel Positive\t|\tLabel Negative\t|")
    print("-----------------------------------------------------------------")
    print("Pred Positive\t|\t{0}\t\t|\t{1}\t".format(buckets[3], buckets[1]))
    print("-----------------------------------------------------------------")
    print("Pred Negative\t|\t{0}\t\t|\t{1}\t".format(buckets[2], buckets[0]))
    print("-----------------------------------------------------------------")

def ExecuteAll(y, yPredicted):
    print(ConfusionMatrix(y, yPredicted))
    print("Accuracy:", Accuracy(y, yPredicted))
    print("Precision:", Precision(y, yPredicted))
    print("Recall:", Recall(y, yPredicted))
    print("FPR:", FalsePositiveRate(y, yPredicted))
    print("FNR:", FalseNegativeRate(y, yPredicted))
    
