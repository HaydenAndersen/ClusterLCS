import pandas as pd
from sklearn.preprocessing import LabelEncoder
from skeLCS import StringEnumerator
import numpy as np


# {'track_accuracy_while_fit':None, 'N':None, 'learning_iterations':None, 'verbose':None, 'p_spec':None,
#                  'movingAvgCount':None, 'chi':None, 'nu':None, 'theta_GA':None, 'n':None}

def sonar():
    data = pd.read_csv('datasets/sonar/sonar.all-data')
    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]
    X = X.to_numpy()
    y = y.to_numpy()
    lb = LabelEncoder()
    y = lb.fit_transform(y)
    params = {'track_accuracy_while_fit':True, 'N':5000, 'learning_iterations':100000, 'verbose':-1, 'p_spec':0.6,
               'movingAvgCount':-1, 'chi':0.8, 'nu':10, 'theta_GA':25, 'n':10, 'discrete_attribute_limit':10}
    return X, y, params

def zoo():
    data = pd.read_csv('datasets/zoo/zoo.data',
                       names=['name', 'hair', 'feathers', 'eggs', 'milk', 'airborne', 'aquatic', 'predator',
                              'toothed', 'backbone', 'breathes', 'venomous', 'fins', 'legs', 'tail', 'domestic',
                              'catsize', 'type'])
    print(data.head())

    data = data.drop(['name'], axis=1)
    X = data.drop(['type'], axis=1).values
    y = data['type'].values
    params = {'track_accuracy_while_fit':True, 'N':150, 'learning_iterations':10000, 'verbose':-1, 'p_spec':0.5,
                 'movingAvgCount':50, 'chi':0.8, 'nu':5, 'theta_GA':25, 'n':6, 'discrete_attribute_limit':10}
    return X, y, params

def wine():
    data = pd.read_csv('datasets/wine/wine.data',
                       names=['class', 'alcohol', 'malic', 'ash', 'alcalinity', 'magnesium', 'phenols', 'flavanoids',
                              'nonflavanoid phenols', 'proanthocyanins', 'color intensity', 'hue', 'OD280/OD315',
                              'proline'])
    print(data.head())

    X = data.drop(['class'], axis=1).values
    y = data['class'].values
    params = {'track_accuracy_while_fit':True, 'N':1000, 'learning_iterations':100000, 'verbose':-1, 'p_spec':0.6,
                 'movingAvgCount':50, 'chi':0.8, 'nu':10, 'theta_GA':25, 'n':5, 'discrete_attribute_limit':20}
    return X, y, params

def mushroom():
    converter = StringEnumerator("datasets/mushroom/agaricus-lepiota.data", "class")
    converter.add_attribute_converter_random('cap-shape')
    converter.add_attribute_converter_random('cap-surface')
    converter.add_attribute_converter_random('cap-color')
    converter.add_attribute_converter_random('bruises')
    converter.add_attribute_converter_random('odor')
    converter.add_attribute_converter_random('gill-attachment')
    converter.add_attribute_converter_random('gill-spacing')
    converter.add_attribute_converter_random('gill-size')
    converter.add_attribute_converter_random('gill-color')
    converter.add_attribute_converter_random('stalk-shape')
    converter.add_attribute_converter_random('stalk-root')
    converter.add_attribute_converter_random('stalk-surface-above-ring')
    converter.add_attribute_converter_random('stalk-surface-below-ring')
    converter.add_attribute_converter_random('stalk-color-above-ring')
    converter.add_attribute_converter_random('stalk-color-below-ring')
    converter.add_attribute_converter_random('veil-type')
    converter.add_attribute_converter_random('veil-color')
    converter.add_attribute_converter_random('ring-number')
    converter.add_attribute_converter_random('ring-type')
    converter.add_attribute_converter_random('spore-print-color')
    converter.add_attribute_converter_random('population')
    converter.add_attribute_converter_random('habitat')
    converter.add_class_converter_random()

    converter.convert_all_attributes()

    headers, classLabel, X, y = converter.get_params()
    params = {'track_accuracy_while_fit':True, 'N':1000, 'learning_iterations':10000, 'verbose':100, 'p_spec':0.5,
                 'movingAvgCount':50, 'chi':0.8, 'nu':5, 'theta_GA':25, 'n':10, 'discrete_attribute_limit':30}
    X[np.where(np.isnan(X))] = 0
    return X, y, params


def get_data(dataset):
    if dataset == 'sonar':
        return sonar()
    if dataset == 'zoo':
        return zoo()
    if dataset == 'wine':
        return wine()
    if dataset == 'mushroom':
        return mushroom()