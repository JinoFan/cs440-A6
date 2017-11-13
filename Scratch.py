import numpy as np
import matplotlib
import neuralnetworks as nn
import mlutils as ml



def trainNNs(X, T, trainFraction, hiddenLayerStructures, numberRepetitions, numberIterations, classify):

    import numpy as np
    import matplotlib
    import neuralnetworks as nn

    results = []

    # debugging
    verbose = False

    for structure in hiddenLayerStructures:
        for i in range(numberRepetitions):
            Xtrain, Ttrain, Xtest, Ttest = ml.partition(X, T, (trainFraction, 1 - trainFraction),
                                                        classification=classify)
            if classify:
                nnet = nn.NeuralNetworkClassifier(1,structure,1)
            else:
                nnet = nn.NeuralNetwork(1,structure,1)

            nnet.train(Xtrain, Ttrain, numberIterations, verbose)

            Y = nnet.use(X)
            Ytest,Ztest = nnet.use(Xtest,allOutputs=True)

            results.append([structure, Xtest, Ztest])

    return results

X = np.arange(10).reshape((-1,1))
T = X + 1 + np.random.uniform(-1, 1, ((10,1)))
results = trainNNs(X, T, 0.8, [2, 10, [10, 10]], 5, 100, classify=False)
print(results)