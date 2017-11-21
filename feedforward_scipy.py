import numpy as np
from Normalize import Normalize
import sklearn
from sklearn.neural_network import MLPRegressor
import pickle

def dump_mlp():
    Ndata = Normalize()
    X = Ndata.inputdata
    Y = Ndata.outputdata

    mlp = MLPRegressor(hidden_layer_sizes=(14, 9),max_iter=12000)
    mlp.fit(X,Y)

    with open("sklearn_nn.txt", "wb") as pickle_file:
        pickle.dump(mlp, pickle_file)

dump_mlp()


# BIN

    # accelerations = []
    # breakings = []
    # steerings = []

    # for y in Y:
    #     accelerations.append(y[0])
    #     breakings.append(y[1])
    #     steerings.append(y[2])

    # mlp_acc = MLPRegressor(hidden_layer_sizes=(14, 9),max_iter=1000)
    # mlp_acc.fit(X,accelerations)
    # mlp_bre = MLPRegressor(hidden_layer_sizes=(14, 9),max_iter=1000)
    # mlp_bre.fit(X,breakings)
    # mlp_ste = MLPRegressor(hidden_layer_sizes=(14, 9),max_iter=100)
    # mlp_ste.fit(X,steerings)