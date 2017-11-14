import numpy as np
import matplotlib
import neuralnetworks as nn



def trainNNs(X, T, trainFraction, hiddenLayerStructures, numberRepetitions, numberIterations, classify=False):

    import numpy as np
    import neuralnetworks as nn
    import time
    import mlutils as ml

    results = []

    # debugging
    verbose = False

    for structure in hiddenLayerStructures:
        trainList = []
        testList = []
        t0 = time.time()
        for i in range(numberRepetitions):
            Xtrain, Ttrain, Xtest, Ttest = ml.partition(X, T, (trainFraction, 1 - trainFraction),
                                                        classification=classify)
            if classify:
                nnet = nn.NeuralNetworkClassifier(X.shape[1],structure,T.shape[1])
            else:
                nnet = nn.NeuralNetwork(X.shape[1],structure,T.shape[1])

            nnet.train(Xtrain, Ttrain, numberIterations)

            Ytrain = nnet.use(Xtrain)
            Ytest,Ztest = nnet.use(Xtest, allOutputs=True)

            trainList.append(np.sqrt(np.mean((Ytrain - Ttrain ) ** 2)))
            testList.append(np.sqrt(np.mean((Ytest - Ttest) ** 2)))

        timeTaken = time.time() - t0
        results.append([structure, trainList, testList, timeTaken])
    return results

def summarize(results):
    import numpy as np
    summary = []
    for item in results:
        summary.append([item[0], np.mean(item[1]), np.mean(item[2]), item[3]])

    return summary

def bestNetwork(summary):
    return min(summary, key = lambda x: x[2])


import pandas as pd

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
results = trainNNs(Xenergy, Tenergy, 0.8, [0, 5, [5, 5], [10, 10]], 10, 100)
print(bestNetwork(summarize(results)))

#print(energydata)
# X = np.arange(10).reshape((-1,1))
# T = X + 1 + np.random.uniform(-1, 1, ((10,1)))
# results = trainNNs(X, T, 0.8, [0, 1, 2], 50, 400, classify=False)
# bestNetwork(summarize(results))