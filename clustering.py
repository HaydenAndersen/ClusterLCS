import pandas as pd
from skeLCS.eLCS import eLCS
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import random
from sklearn_extra.cluster import KMedoids
from hdbscan import HDBSCAN
import argparse
import json
import time
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.neighbors import KNeighborsClassifier
from clustering_utils import *
from read_data import get_data
from skrebate.surf import SURF
from functools import partial
from sklearn.feature_selection import SelectKBest



class clusterLCS(eLCS):
    def __init__(self, learning_iterations=10000, track_accuracy_while_fit = False, N=1000, p_spec=0.5, discrete_attribute_limit=10,
                 nu=5, chi=0.8, mu=0.04, theta_GA=25, theta_del=20, theta_sub=20,
                 acc_sub=0.99, beta=0.2, delta=0.1, init_fit=0.01, fitness_reduction=0.1, do_correct_set_subsumption=False,
                 do_GA_subsumption=True, selection_method='tournament', theta_sel=0.5, random_state = None,match_for_missingness=False,
                 reboot_filename=None, verbose=0, movingAvgCount=50, method='kmeans', n=10, choice='all', outliers=True, distance='euclidean'):
        self.method = method
        self.n = n
        self.choice = choice
        self.outliers = outliers
        self.distance = distance
        super(clusterLCS, self).__init__(learning_iterations=learning_iterations,track_accuracy_while_fit =track_accuracy_while_fit ,N=N,p_spec=p_spec,discrete_attribute_limit=discrete_attribute_limit,nu=nu,chi=chi,mu=mu,theta_GA=theta_GA,theta_del=theta_del,theta_sub=theta_sub,acc_sub=acc_sub,beta=beta,delta=delta,init_fit=init_fit,fitness_reduction=fitness_reduction,do_correct_set_subsumption=do_correct_set_subsumption,do_GA_subsumption=do_GA_subsumption,selection_method=selection_method,theta_sel=theta_sel,random_state =random_state ,match_for_missingness=match_for_missingness,reboot_filename=reboot_filename,verbose=verbose,movingAvgCount=movingAvgCount)

    def __kmeans(self, features, n, seed):
        kmeans = KMeans(n, random_state=seed)
        kmeans.fit(features)
        dists = kmeans.transform(features)
        sel = []
        for i in range(n):
            thisclust = dists[:, i]
            ind = np.argmin(thisclust)
            sel.append(ind)
        X_selected = features.T[:, sel]
        return X_selected, sel

    def __kmedoids(self, features, n, seed, matrix=None):
        if matrix is not None:
            kmedoids = KMedoids(n, random_state=seed, metric='precomputed')
            kmedoids.fit(matrix)
        else:
            kmedoids = KMedoids(n, random_state=seed)
            kmedoids.fit(features)
        sel = kmedoids.medoid_indices_
        print(sel)
        X_selected = features.T[:, sel]
        return X_selected, sel

    def euclidean(self, a, b):
        ret = np.sqrt(np.sum(np.square(np.subtract(a, b)), 1))
        return ret

    def __hdbscan(self, features, center, outliers, seed, matrix=None):
        if center not in {'all', 'single', 'geometric', 'wgeometric'}:
            raise ValueError("Chosen method '{}' doesn't exist".format(center))
        if matrix is not None:
            hdbscan = HDBSCAN(metric='precomputed')
            hdbscan.fit(matrix)
        else:
            hdbscan = HDBSCAN()
            hdbscan.fit(features)
        # hdbscan = HDBSCAN()
        # hdbscan.fit(features)
        membership = hdbscan.probabilities_
        labels = hdbscan.labels_
        num_clusters = max(labels) + 1
        if outliers:
            sel = [i for i, l in enumerate(labels) if l == -1]
        else:
            sel = []
        print(sel)
        print(num_clusters)
        for n in range(num_clusters):
            indexes = [i for i, m in enumerate(membership) if labels[i] == n and m==1]
            if center == 'all':
                sel.extend(indexes)
            elif center == 'single':
                if seed:
                    random.seed(seed)
                sel.append(random.choice(indexes))
            elif center == 'geometric' or center == 'wgeometric':
                cluster = [i for i, l in enumerate(labels) if l == n]
                cluster_features = features[cluster]
                num_feat = len(cluster)
                if center == 'geometric':
                    mean = np.mean(cluster_features, axis=0)
                elif center == 'wgeometric':
                    mean = np.average(cluster_features, axis=0, weights=membership[cluster])
                    print(mean)
                else:
                    raise NotImplementedError('Should not be able to reach here')
                mean_rep = np.tile(mean, [num_feat,1])
                dist = self.euclidean(mean_rep, cluster_features)
                min_index = np.argmin(dist)
                best = cluster[min_index]
                sel.append(best)
        X_selected = features.T[:, sel]
        return X_selected, sel

    def clustered_features(self, X, matrix=None):
        seed = self.random_state
        method = self.method
        n = self.n
        choice = self.choice
        outliers = self.outliers
        if method not in {'kmeans', 'kmedoids', 'hdbscan', 'baseline'}:
            raise ValueError("Chosen method '{}' doesn't exist".format(method))
        if method == 'kmeans':
            features = X.T
            return self.__kmeans(features, n, seed)
        if method == 'kmedoids':
            features = X.T
            return self.__kmedoids(features, n, seed, matrix)
        if method == 'hdbscan':
            features = X.T
            return self.__hdbscan(features, choice, outliers, seed, matrix)


    def fit(self, X, y):
        if self.distance == 'euclidean':
            X_selected, sel = self.clustered_features(X)
        elif self.distance == 'mi':
            attributeDiscrete = get_discrete(X, discrete_attribute_limit=self.discrete_attribute_limit)
            matrix = get_matrix(X, attributeDiscrete)
            assert (matrix == matrix.T).all()
            X_selected, sel = self.clustered_features(X, matrix)
        else:
            raise NotImplementedError('Chosen distance function is not implemented')
        self.sel_ = sel
        return super(clusterLCS, self).fit(X_selected, y)

    def predict_proba(self, X):
        X_selected = X[:, self.sel_]
        return super(clusterLCS, self).predict_proba(X_selected)

    def predict(self, X):
        X_selected = X[:, self.sel_]
        return super(clusterLCS, self).predict(X_selected)


def main(args):
    # data = pd.read_csv('datasets/sonar/sonar.all-data')
    # X = data.iloc[:,:-1]
    # y = data.iloc[:,-1]
    # X = X.to_numpy()
    # y = y.to_numpy()
    # lb = LabelEncoder()
    # y = lb.fit_transform(y)

    X, y, params = get_data(args.dataset)
    if args.method == 'surf':
        surf = SURF(n_features_to_select=params['n'], discrete_threshold=params['discrete_attribute_limit'])
        X = surf.fit_transform(X, y)
    if args.method == 'baseselect':
        attributeDiscrete = get_discrete(X, discrete_attribute_limit=params['discrete_attribute_limit'])
        scorer = partial(ent, attributeDiscrete=attributeDiscrete)
        topK = SelectKBest(scorer, k=params['n'])
        X = topK.fit_transform(X, y)
    # from nbclust import get_num_clusters
    # n = get_num_clusters(X.T)
    # print(n)
    # raise ValueError
    if args.method == 'baseline' or args.method == 'surf' or args.method == 'baseselect':
        lcs = eLCS(track_accuracy_while_fit=params['track_accuracy_while_fit'], N=params['N'], learning_iterations=params['learning_iterations'],
                   verbose=params['verbose'], p_spec=params['p_spec'], movingAvgCount=params['movingAvgCount'],
               chi=params['chi'], nu=params['nu'], theta_GA=params['theta_GA'], random_state=args.seed, discrete_attribute_limit=params['discrete_attribute_limit'], theta_sel=0.3)
    else:
        lcs = clusterLCS(track_accuracy_while_fit=params['track_accuracy_while_fit'], N=params['N'], learning_iterations=params['learning_iterations'],
                   verbose=params['verbose'], p_spec=params['p_spec'], movingAvgCount=params['movingAvgCount'],
               chi=params['chi'], nu=params['nu'], theta_GA=params['theta_GA'], random_state=args.seed, method=args.method,
                         choice=args.choice, outliers=args.outliers, distance=args.distance, discrete_attribute_limit=params['discrete_attribute_limit'],
                         n=params['n'], theta_sel=0.3)


    start = time.time()
    # lcs = eLCS(track_accuracy_while_fit=True, N=500, learning_iterations=1000000, verbose=-1, p_spec=0.6,
    #            movingAvgCount=-1, chi=1.0, nu=2, theta_GA=15, random_state=args.seed)

    scores = cross_val_score(estimator=lcs, X=X, y=y, cv=10)
    time_taken = time.time() - start



    print(scores)
    print('{} +- {}'.format(scores.mean(), scores.std()*2))
    # print('Selected features: {}'.format(sel))
    print('Time taken: {}'.format(time_taken))
    values = {'scores':list(scores), 'mean':scores.mean(), 'std':scores.std(), 'time':time_taken}
    with open('output/{}-{}-{}{}{}-{}.json'.format(
            args.dataset,
            args.method,
            args.distance,
            '-'+args.choice if args.method == 'hdbscan' else '',
            '-' + str(args.outliers) if args.method == 'hdbscan' else '',
            args.seed
    ), 'w') as f:
        json.dump(values, f)

    # lcs.fit(X, y)
    # print(lcs.score(X, y))
    # header = np.array(['alcohol', 'malic', 'ash', 'alcalinity', 'magnesium', 'phenols', 'flavanoids',
    #                    'nonflavanoid phenols', 'proanthocyanins', 'color intensity', 'hue', 'OD280/OD315',
    #                    'proline'])
    # lcs.export_final_rule_population(header, [i for i in range(1,4)], 'output/{}-{}-{}{}{}-{}.csv'.format(
    #         args.dataset,
    #         args.method,
    #         args.distance,
    #         '-'+args.choice if args.method == 'hdbscan' else '',
    #         '-' + str(args.outliers) if args.method == 'hdbscan' else '',
    #         args.seed
    # ), False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--method', default='baseline')
    parser.add_argument('-c', '--choice', default='all')
    parser.add_argument('-o', '--outliers', action='store_true')
    parser.add_argument('-dis', '--distance', default='euclidean')
    parser.add_argument('-d', '--dataset', default='sonar')
    parser.add_argument('-s', '--seed', type=int, default=1)
    args = parser.parse_args()
    main(args)
