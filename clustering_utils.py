import numpy as np
from sklearn.preprocessing import OneHotEncoder
from entropy_estimators import entropy, entropyd, centropy, centropyd, centropycd, centropydc, mi, midd, micd, midc
from itertools import combinations_with_replacement

def get_discrete(X, discrete_attribute_limit=10):
    numAttributes = X.shape[1]
    numTrainInstances = X.shape[0]
    attributeDiscrete = np.empty(numAttributes, np.bool)
    for att in range(numAttributes):
        attIsDiscrete = True
        currentInstanceIndex = 0
        stateDict = {}
        while attIsDiscrete and len(list(
                stateDict.keys())) <= discrete_attribute_limit and currentInstanceIndex < numTrainInstances:
            target = X[currentInstanceIndex, att]
            if target in list(stateDict.keys()):
                stateDict[target] += 1
            elif np.isnan(target):
                pass
            else:
                stateDict[target] = 1
            currentInstanceIndex += 1

        if len(list(stateDict.keys())) > discrete_attribute_limit:
            attIsDiscrete = False
        if attIsDiscrete:
            attributeDiscrete[att] = True
        else:
            attributeDiscrete[att] = False
    return attributeDiscrete

def get_continuous(X, discrete_attribute_limit=10):
    return np.invert(get_discrete(X, discrete_attribute_limit))

def VI(x, y, x_disc, y_disc):
    # if not x_disc or not y_disc:
    #     print(x_disc, y_disc)
    #     print('X', set(x))
    #     print('Y', set(y))
    x, y = np.asarray(x), np.asarray(y)
    x, y = x.reshape(x.shape[0], -1), y.reshape(y.shape[0], -1)
    if x_disc:
        if y_disc:
            alg1 = centropyd
            alg2 = centropyd
        else:
            alg1 = centropydc
            alg2 = centropycd
    else:
        if y_disc:
            alg1 = centropycd
            alg2 = centropydc
        else:
            alg1 = centropy
            alg2 = centropy
    return alg1(x, y) + alg2(y, x)

def get_matrix(X, attributeDiscrete):
    features = X.T
    indexes = [i for i in range(len(features))]
    feature_indexes = list(combinations_with_replacement(indexes, 2))
    matrix = np.zeros((len(features), len(features)))
    # print(matrix)
    for xindex, yindex in feature_indexes:
        x = features[xindex]
        y = features[yindex]
        x_disc = attributeDiscrete[xindex]
        y_disc = attributeDiscrete[yindex]
        res = VI(x, y, x_disc, y_disc)
        matrix[xindex, yindex] = res
        matrix[yindex, xindex] = res
    matrix[matrix < 0] = 0
    return matrix

def ent(X, y, attributeDiscrete):
    features = X.T
    scores = np.zeros(X.shape[1])
    for i, feature in enumerate(features):
        f = feature.reshape(feature.shape[0], -1)
        if attributeDiscrete[i]:
            scores[i] = entropyd(f)
        else:
            scores[i] = entropy(f)
    return scores