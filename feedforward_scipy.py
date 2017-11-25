import numpy as np
from Normalize import Normalize
import sklearn
from sklearn.neural_network import MLPRegressor
import pickle

def dump_mlp():
    Ndata = Normalize()
    X = Ndata.inputdata
    Y = Ndata.outputdata
    for x in X:
        x = [x[0],x[1],x[2],x[11],x[12],x[13]]

    mlp = MLPRegressor(hidden_layer_sizes=(200,15),max_iter=1200, learning_rate='adaptive')
    mlp.fit(X,Y)

    T = mlp.predict(X)
    error = 0
    for i, t in enumerate(T):
        error += abs(round(t[0])-Y[i][0]) + abs(round(t[1])-Y[i][1]) + abs(t[2]-Y[i][2])
    print (error)

    with open("sklearn_nn.txt", "wb") as pickled:
        pickle.dump(mlp, pickled)

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