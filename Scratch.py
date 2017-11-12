import numpy as np
import matplotlib
import neuralnetworks as nn
import mlutils as ml

nnet = nn.NeuralNetworkClassifier

nnet.train(X,T,)



def trainNNs(X, T, trainFraction, hiddenLayerStructures, numberRepetitions, numberIterations, classify):

    import numpy as np
    import matplotlib
    import neuralnetworks as nn

    # debugging
    verbose = True

    for structure in hiddenLayerStructures:
        Xtrain, Ttrain, Xvalidate, Tvalidate, Xtest, Ttest = partition(X, T, (0.6, 0.2, 0.2), classification=True)
        for i in range(numberRepetitions):
            if classify:
                nnet = nn.NeuralNetworkClassifier
            else:
                nnet = nn.NeuralNetwork

            nnet.train(X, T, numberIterations, verbose)


results = trainNNs(X, T, trainFraction, hiddenLayerStructures, numberRepetitions, numberIterations, classify)