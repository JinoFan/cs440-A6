import numpy as np
import matplotlib
import neuralnetworks as nn



def trainNNs(X, T, trainFraction, hiddenLayerStructures, numberRepetitions, numberIterations, classify=False):
    """
    Trains neural networks repeatedly.
    :param X: Data to partition and train
    :param T: Target values
    :param trainFraction: What percent of the data should be used for training
    :param hiddenLayerStructures: Number of hidden layer structures while training
    :param numberRepetitions: Number of times to run train
    :param numberIterations: Iterations within Neural Network
    :param classify: Classification or Regression
    :return: List containing the hidden layer structure, the training error and testing error, and the elapsed time.
    """
    import numpy as np
    import neuralnetworks as nn
    import time
    import mlutils as ml

    results = []
    global resultErrors
    resultErrors = []

    # debugging
    verbose = True

    for structure in hiddenLayerStructures:
        trainList = []
        testList = []
        t0 = time.time()
        for i in range(numberRepetitions):
            Xtrain, Ttrain, Xtest, Ttest = ml.partition(X, T, (trainFraction, 1 - trainFraction), classification=classify)
            if classify:
                nnet = nn.NeuralNetworkClassifier(X.shape[1],structure,len(np.unique(T)))
            else:
                nnet = nn.NeuralNetwork(X.shape[1],structure,T.shape[1])
sa
            nnet.train(Xtrain, Ttrain, numberIterations)

            Ytrain = nnet.use(Xtrain)

            Ytest = nnet.use(Xtest)

            trainList.append(np.sqrt(np.mean((Ytrain - Ttrain ) ** 2)))
            testList.append(np.sqrt(np.mean((Ytest - Ttest) ** 2)))

        timeTaken = time.time() - t0
        resultErrors.append([structure, nnet.getErrorTrace()])
        results.append([structure, trainList, testList, timeTaken])
    return results

def summarize(results):
    """
    Summarizes the return data from trainNNs
    :param results: List containing the hidden layer structure, the training error and testing error, and the elapsed time.
    :return: List returning the mean errors for the iterations within trainNNs
    """
    import numpy as np
    summary = []
    for item in results:
        summary.append([item[0], np.mean(item[1]), np.mean(item[2]), item[3]])

    return summary

def bestNetwork(summary):
    """
    Returns the best hidden structure/error/time
    :param summary: Summary data from summarize
    :return: Best hidden structure/error/time from the list.
    """
    return min(summary, key = lambda x: x[2])


import pandas as pd

# frogdata = pd.read_csv("Frogs_MFCCs.csv")
# print(frogdata.columns.values)
#
# speciesList = frogdata["Species"]
#
# uniqueSpecies = np.unique(speciesList)
#
# speciesDict = dict(enumerate(uniqueSpecies))
#
# Tanuran = speciesList.replace(speciesDict.values(), speciesDict.keys()).as_matrix().reshape(-1,1)
# Xanuran = frogdata.drop(["MFCCs_ 1","Family", "Genus", "Species", "RecordID"], axis=1).as_matrix()
#
# print(Xanuran.shape)
# print(Tanuran.shape)
# print(Xanuran[:2,:])
# print(Tanuran[:2])
#
# for i in range(10):
#     print('{} samples in class {}'.format(np.sum(Tanuran==i), i))
#
# results = trainNNs(Xanuran, Tanuran, 0.8, [0, 3,  5, [5, 5], [10,10,10]], 5, 100, classify=True)
#
# print(summarize(results))
#
# print(bestNetwork(summarize(results)))
energydata = pd.read_csv("energydata_complete.csv")
energydata = energydata.drop(["date","rv1","rv2"], axis=1)

print(energydata.columns.values)

# print(energydata.as_matrix().shape)
# print(energydata.as_matrix()[:2,:])

Tenergy = energydata[["Appliances","lights"]].as_matrix()
#print(Tenergy)

Xenergy = energydata.drop(["Appliances", "lights"], axis=1).as_matrix()

print(Tenergy.shape)
print(Xenergy.shape)
results = trainNNs(Xenergy, Tenergy, 0.8, [0, 1, 2, 3, 4, 5, [5, 5], [10, 10], [5,5,5]], 10, 100)
print(summarize(results))
print(bestNetwork(summarize(results)))

#print(energydata)
# X = np.arange(10).reshape((-1,1))
# T = X + 1 + np.random.uniform(-1, 1, ((10,1)))
# results = trainNNs(X, T, 0.8, [0, 1, 2], 50, 400, classify=False)
# bestNetwork(summarize(results))