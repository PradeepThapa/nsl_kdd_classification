# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 18:08:02 2020

"""

import os
import timeit
import warnings
from collections import defaultdict

import catboost as cb
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.under_sampling import CondensedNearestNeighbour
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, \
    plot_confusion_matrix
from sklearn.metrics import confusion_matrix, zero_one_loss
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from termcolor import colored

# import imblearn

warnings.filterwarnings('ignore')

np.random.seed(100)

dataset_root = '/Users/pradeep/PycharmProjects/Week5HD/Data/NSL-KDD-Dataset'

train_file = os.path.join(dataset_root, 'KDDTrain+.txt')
test_file = os.path.join(dataset_root, 'KDDTest+.txt')

# Original KDD dataset feature names obtained from
# http://kdd.ics.uci.edu/databases/kddcup99/kddcup.names
# http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html

header_names = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
                'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
                'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
                'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
                'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
                'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
                'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type',
                'success_pred']

# Differentiating between nominal, binary, and numeric features

# root_shell is marked as a continuous feature in the kddcup.names 
# file, but it is supposed to be a binary feature according to the 
# dataset documentation

col_names = np.array(header_names)

nominal_idx = [1, 2, 3]
binary_idx = [6, 11, 13, 14, 20, 21]
numeric_idx = list(set(range(41)).difference(nominal_idx).difference(binary_idx))

nominal_cols = col_names[nominal_idx].tolist()
binary_cols = col_names[binary_idx].tolist()
numeric_cols = col_names[numeric_idx].tolist()

# training_attack_types.txt maps each of the 22 different attacks to 1 of 4 categories
# file obtained from http://kdd.ics.uci.edu/databases/kddcup99/training_attack_types


category = defaultdict(list)
category['benign'].append('normal')

with open('/Users/pradeep/PycharmProjects/Week5HD/Data/NSL-KDD-Dataset/training_attack_types.txt', 'r') as f:
    for line in f.readlines():
        attack, cat = line.strip().split(' ')
        category[cat].append(attack)

attack_mapping = dict((v, k) for k in category for v in category[k])

train_df = pd.read_csv(train_file, names=header_names)

train_df['attack_category'] = train_df['attack_type'] \
    .map(lambda x: attack_mapping[x])
train_df.drop(['success_pred'], axis=1, inplace=True)

test_df = pd.read_csv(test_file, names=header_names)
test_df['attack_category'] = test_df['attack_type'] \
    .map(lambda x: attack_mapping[x])
test_df.drop(['success_pred'], axis=1, inplace=True)

train_attack_types = train_df['attack_type'].value_counts()
train_attack_cats = train_df['attack_category'].value_counts()

test_attack_types = test_df['attack_type'].value_counts()
test_attack_cats = test_df['attack_category'].value_counts()

train_attack_types.plot(kind='barh', figsize=(20, 10), fontsize=20)

train_attack_cats.plot(kind='barh', figsize=(20, 10), fontsize=30)

test_attack_types.plot(kind='barh', figsize=(20, 10), fontsize=15)

test_attack_cats.plot(kind='barh', figsize=(20, 10), fontsize=30)

# Let's take a look at the binary features
# By definition, all of these features should have a min of 0.0 and a max of 1.0
# execute the commands in console

train_df[binary_cols].describe().transpose()

# Wait a minute... the su_attempted column has a max value of 2.0?

train_df.groupby(['su_attempted']).size()

# Let's fix this discrepancy and assume that su_attempted=2 -> su_attempted=0

train_df['su_attempted'].replace(2, 0, inplace=True)
test_df['su_attempted'].replace(2, 0, inplace=True)
train_df.groupby(['su_attempted']).size()

# Next, we notice that the num_outbound_cmds column only takes on one value!

train_df.groupby(['num_outbound_cmds']).size()

# Now, that's not a very useful feature - let's drop it from the dataset

train_df.drop('num_outbound_cmds', axis=1, inplace=True)
test_df.drop('num_outbound_cmds', axis=1, inplace=True)
numeric_cols.remove('num_outbound_cmds')

"""
Data Preparation

"""
train_Y = train_df['attack_category']
train_x_raw = train_df.drop(['attack_category', 'attack_type'], axis=1)
test_Y = test_df['attack_category']
test_x_raw = test_df.drop(['attack_category', 'attack_type'], axis=1)

'''# feature selection
def select_features(X_train, y_train, X_test):
    fs = SelectKBest(score_func=mutual_info_classif, k=30)
    fs.fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs
'''

combined_df_raw = pd.concat([train_x_raw, test_x_raw])
combined_df = pd.get_dummies(combined_df_raw, columns=nominal_cols, drop_first=True)

train_x = combined_df[:len(train_x_raw)]
test_x = combined_df[len(train_x_raw):]

# use this for catboost
x_train = train_x_raw
x_test = test_x_raw

# Store dummy variable feature names
dummy_variables = list(set(train_x) - set(combined_df_raw))

# execute the commands in console
train_x.describe()
train_x['duration'].describe()

# Experimenting with StandardScaler on the single 'duration' feature
durations = train_x['duration'].values.reshape(-1, 1)
standard_scaler = StandardScaler().fit(durations)
scaled_durations = standard_scaler.transform(durations)
pd.Series(scaled_durations.flatten()).describe()

# Experimenting with MinMaxScaler on the single 'duration' feature

min_max_scaler = MinMaxScaler().fit(durations)
min_max_scaled_durations = min_max_scaler.transform(durations)
pd.Series(min_max_scaled_durations.flatten()).describe()

# Experimenting with RobustScaler on the single 'duration' feature

min_max_scaler = RobustScaler().fit(durations)
robust_scaled_durations = min_max_scaler.transform(durations)
pd.Series(robust_scaled_durations.flatten()).describe()

# Let's proceed with StandardScaler- Apply to all the numeric columns

standard_scaler = StandardScaler().fit(train_x[numeric_cols])

train_x[numeric_cols] = \
    standard_scaler.transform(train_x[numeric_cols])

test_x[numeric_cols] = \
    standard_scaler.transform(test_x[numeric_cols])

train_x.describe()

train_Y_bin = train_Y.apply(lambda x: 0 if x is 'benign' else 1)
test_Y_bin = test_Y.apply(lambda x: 0 if x is 'benign' else 1)

'''# transform the dataset
oversample = SMOTE()
train_x, train_Y = oversample.fit_resample(train_x, train_Y)'''

"""
multi class classification using decision tree 

"""


# Decision Tree
def decision_tree_clf():
    print("------Decision Tree Classification-------")

    # build Decision Tree classifier
    classifier = DecisionTreeClassifier(random_state=17)

    # Train Classifier
    classifier.fit(train_x, train_Y)

    # predict
    pred_y = classifier.predict(test_x)

    # confusion matrix
    results = confusion_matrix(test_Y, pred_y)

    # error rate
    error = zero_one_loss(test_Y, pred_y)

    # print results
    print(results)
    print(error)


# answers to question 4
def data_exploration_solution():
    print("*******************")
    print("Step 4: Data Exploration (Understanding the data)")
    print("*******************")
    print("1. Identify the attribute names (Header)")
    print(train_df.columns)
    print("2. Check the length of the Train and Test dataset")
    print("length of Train dataset: ", train_df.size)
    print("length of Test dataset: ", test_df.size)
    print("3. Check the total number of samples that belong to each of the five classes of the training dataset.")
    print(train_df.groupby('attack_category')['flag'].count())
    print("*******************")


# random forest using hperparameter tuning
def random_forest_grid_search():
    # Creating a grid of different hyperparameters
    grid_params = {
        'n_estimators': [60],
        'criterion': ["gini"],
        'min_samples_split': [2, 4, 6, 10],
        'max_depth': [20, 25, 30],
        # 'max_leaf_nodes': [1, 5, 7, 10]
    }

    # random forest classifer
    clf = RandomForestClassifier()

    print("Searching for optimal parameters..............")

    # Building a 3 fold Cross-Validated GridSearchCV object
    grid_object = GridSearchCV(estimator=clf, param_grid=grid_params, cv=3)

    print("Training the data...............")

    # Fitting the grid to the training data
    grid_object.fit(train_x, train_Y)

    # Extracting the best parameters
    print(grid_object.best_params_)

    # Extracting the best model
    rf_best = grid_object.best_estimator_
    print(rf_best)

    print(grid_object.best_score_)


# Random Forest
def random_forest_clf():
    print(colored("------Random Forest Classification-------", 'red'))
    # build classifier
    clf = RandomForestClassifier(criterion='entropy', max_depth=30, n_estimators=48, random_state=0)

    # start timer
    start_time = timeit.default_timer()

    print("Training the Random Forest Classifier.......")

    clf = clf.fit(train_x, train_Y)

    # end timer
    print("The time difference is :", timeit.default_timer() - start_time)

    print("Predicting test data.......")

    '''features = clf.feature_importances_
    feature_cols = []
    # print feature importance
    for i, j in enumerate(features, 1):
        if j <= 0.0:
            feature_cols.append(i)
            print(i)
    new_train_x = train_x.copy()
    new_test_x = test_x.copy()

    for k in feature_cols:
        new_train_x.drop(new_train_x.columns[k], axis=1, inplace=True)
        new_test_x.drop(new_test_x.columns[k], axis=1, inplace=True)

    clf2 = RandomForestClassifier(n_estimators=240, random_state=0)
    clf2 = clf2.fit(new_train_x, train_Y)'''

    # predict test data
    pred_y = clf.predict(test_x)

    # analyse prediction
    c_matrix = confusion_matrix(test_Y, pred_y)  # confusion matrix
    error = zero_one_loss(test_Y, pred_y)  # error
    score = accuracy_score(test_Y, pred_y)  # accuracy score

    print('Confusion Matrix\n---------------------------\n', c_matrix)
    print('---------------------------')
    print("Error: {:.4f}%".format(error * 100))
    print("Accuracy Score: {:.4f}%".format(score * 100))
    print(classification_report(test_Y, pred_y))
    print('accuracy: ', c_matrix.diagonal() / c_matrix.sum(axis=1))

    # Plot non-normalized confusion matrix
    plot_confusion_matrix(clf, test_x, test_Y, cmap=plt.cm.Greens, values_format='.0f', xticks_rotation='horizontal')
    plt.title("Confusion Matrix for Random Forest")
    plt.show()


# Hyperparameter tuning for KNN
def knn_grid_search():
    # parameters
    grid_params = {
        'n_neighbors': [2, 5, 7, 10, 12],
        'leaf_size': [10, 20, 30, 50, 100]
    }

    # KNN classifier
    clf = KNeighborsClassifier(n_jobs=-1)

    print("Searching for optimal parameters..............")

    # Building a 3 fold Cross-Validated GridSearchCV object
    grid_object = GridSearchCV(estimator=clf, param_grid=grid_params, cv=10)

    print("Training the data...............")

    # Fitting the grid to the training data
    grid_object.fit(train_x, train_Y)

    # Extracting the best parameters
    print(grid_object.best_params_)

    # Extracting the best model
    rf_best = grid_object.best_estimator_
    print(rf_best)


# Find best K value in KNN
def find_k():
    # find value for parameter n_neighbors value between 1 to 20 where accuracy is higher
    for i in range(1, 21):
        # KNN classifier
        clf_knn = KNeighborsClassifier(n_neighbors=i)

        # train data
        clf_knn = clf_knn.fit(train_x, train_Y)

        # predict
        pred_y = clf_knn.predict(test_x)

        print('accuracy for k value ', i, ': ', accuracy_score(test_Y, pred_y))


# KNN classifier
def knn_clf():
    print(colored("------KNN Classification-------", 'red'))

    # KNN classifier
    clf_knn = KNeighborsClassifier(n_neighbors=7)  # using 7 because it has higher accuray rate

    # start timer
    starttime = timeit.default_timer()

    print("Training the KNN Classifier.......")

    # Train model
    clf_knn = clf_knn.fit(train_x, train_Y)

    print("The time difference is :", timeit.default_timer() - starttime)

    print("Predicting test data.......")

    # predict
    pred_y = clf_knn.predict(test_x)

    # analyse results
    c_matrix = confusion_matrix(test_Y, pred_y)  # confusion matrix
    error = zero_one_loss(test_Y, pred_y)  # error
    score = accuracy_score(test_Y, pred_y)

    # display results
    print('Confusion Matrix\n---------------------------\n', c_matrix)
    print('---------------------------')
    print("Error: {:.4f}%".format(error * 100))
    print("Accuracy Score: {:.4f}%".format(score * 100))
    print(classification_report(test_Y, pred_y))
    print('accuracy: ', c_matrix.diagonal() / c_matrix.sum(axis=1))

    # Plot non-normalized confusion matrix
    disp = plot_confusion_matrix(clf_knn, test_x, test_Y, cmap=plt.cm.Greens, values_format='.0f',
                                 xticks_rotation='horizontal')
    plt.title("Confusion Matrix for k-nearest neighbors")

    plt.show()


# SVM classification
def svc_grid_search():
    # Creating a grid of different hyperparameters
    grid_params = {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
        'gamma': ['scale', 'auto']
    }

    # SVC estimator
    clf = SVC(random_state=0)

    print("Searching for optimal parameters..............")

    # Building a 3 fold Cross-Validated GridSearchCV object
    grid_object = GridSearchCV(estimator=clf, param_grid=grid_params, cv=10)

    print("Training the data...............")

    # Fitting the grid to the training data
    grid_object.fit(train_x, train_Y)

    # Extracting the best parameters
    print(grid_object.best_params_)

    # Extracting the best model
    rf_best = grid_object.best_estimator_
    print("Best Parameters are:\n", rf_best)


# SVC
def svm_clf():
    print(colored("------SVM Classification-------", 'red'))
    # build classifier
    clf_svc = SVC(kernel='poly', degree=1, C=3)  # using poly for kernel

    # start timer
    starttime = timeit.default_timer()

    print("Training the SVM Classifier.......")

    # train SVC
    clf_svc = clf_svc.fit(train_x, train_Y)

    print("The time difference is :", timeit.default_timer() - starttime)

    print("Predicting test data.......")

    # predict
    pred_y = clf_svc.predict(test_x)

    # anlayse results
    c_matrix = confusion_matrix(test_Y, pred_y)
    error = zero_one_loss(test_Y, pred_y)
    score = accuracy_score(test_Y, pred_y)

    # display results
    print('Confusion Matrix\n---------------------------\n', c_matrix)
    print('---------------------------')
    print("Error: {:.4f}%".format(error * 100))
    print("Accuracy Score: {:.4f}%".format(score * 100))
    print(classification_report(test_Y, pred_y))
    print('accuracy: ', c_matrix.diagonal() / c_matrix.sum(axis=1))

    # Plot non-normalized confusion matrix
    disp = plot_confusion_matrix(clf_svc, test_x, test_Y, cmap=plt.cm.Greens, values_format='.0f',
                                 xticks_rotation='horizontal')
    plt.title("Confusion Matrix for SVM")

    plt.show()


# logistic regression hyperparameter tuning
def logistic_reg_grid_search():
    # Creating a grid of different hyperparameters
    grid_params = {
        'penalty': ['l1', 'l2'],
        'max_iter': [100, 200, 300, 500, 800, 1000]
    }

    # logistic regression classifier
    clf = LogisticRegression(random_state=0)

    print("Searching for optimal parameters..............")

    # Building a 10 fold Cross-Validated GridSearchCV object
    grid_object = GridSearchCV(estimator=clf, param_grid=grid_params, cv=10)

    print("Training the model...............")

    # Fitting the grid to the training data
    grid_object.fit(train_x, train_Y)

    # Extracting the best parameters
    print(grid_object.best_params_)

    # Extracting the best model
    rf_best = grid_object.best_estimator_
    print(rf_best)


# Logistic Regression
def logistic_reg_clf():
    print(colored("------Logistic Regression Classification-------", 'red', attrs='bold'))
    # logistic regression classifier
    clf_lr = LogisticRegression(C=1e5, random_state=0)

    # start timer
    starttime = timeit.default_timer()

    print("Training the Logistic Regression Classifier.......")

    # train the model
    clf_lr = clf_lr.fit(train_x, train_Y)

    print("The time difference is :", timeit.default_timer() - starttime)

    print("Predicting test data.......")

    # predict
    pred_y = clf_lr.predict(test_x)

    # get results
    c_matrix = confusion_matrix(test_Y, pred_y)
    error = zero_one_loss(test_Y, pred_y)
    score = accuracy_score(test_Y, pred_y)

    # display results
    print('Confusion Matrix\n---------------------------\n', c_matrix)
    print('---------------------------')
    print("Error: {:.4f}%".format(error * 100))
    print("Accuracy Score: {:.4f}%".format(score * 100))
    print(classification_report(test_Y, pred_y))
    print('accuracy: ', c_matrix.diagonal() / c_matrix.sum(axis=1))

    # Plot non-normalized confusion matrix
    disp = plot_confusion_matrix(clf_lr, test_x, test_Y, cmap=plt.cm.Greens, values_format='.0f',
                                 xticks_rotation='horizontal')
    plt.title("Confusion Matrix for Logistic Regression")

    plt.show()


# hyperparameter tuning for SGD
def sgd_grid_search():
    # Creating a grid of different hyperparameters
    grid_params = {
        'loss': ['hinge', 'log'],
        'penalty': ['l2', 'l1'],
        'max_iter': [100, 200, 300, 400, 500],
        'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive']
    }

    # SGD classifier
    clf = SGDClassifier(random_state=0)

    print("Searching for optimal parameters..............")

    # Building a 10 fold Cross-Validated GridSearchCV object
    grid_object = GridSearchCV(estimator=clf, param_grid=grid_params, cv=10)

    print("Training the model...............")

    # Fitting the grid to the training data
    grid_object.fit(train_x, train_Y)

    # Extracting the best parameters
    print(grid_object.best_params_)

    # Extracting the best model
    rf_best = grid_object.best_estimator_
    print(rf_best)


# SGD classification
def sgd_clf():
    print(colored("------SGD Classification-------", 'red'))
    # build classifier
    clf_sgd = SGDClassifier(loss="hinge", penalty="l1", max_iter=200, alpha=0.001, random_state=0)

    # start timer
    starttime = timeit.default_timer()

    print("Training the SGD Classifier.......")

    # train model
    clf_sgd = clf_sgd.fit(train_x, train_Y)

    print("The time difference is :", timeit.default_timer() - starttime)

    print("Predicting test data.......")

    # predict
    pred_y = clf_sgd.predict(test_x)

    c_matrix = confusion_matrix(test_Y, pred_y)
    error = zero_one_loss(test_Y, pred_y)
    score = accuracy_score(test_Y, pred_y)

    print('Confusion Matrix\n---------------------------\n', c_matrix)
    print('---------------------------')
    print("Error: {:.4f}%".format(error * 100))
    print("Accuracy Score: {:.4f}%".format(score * 100))
    print(classification_report(test_Y, pred_y))
    print('accuracy: ', c_matrix.diagonal() / c_matrix.sum(axis=1))

    # Plot non-normalized confusion matrix
    disp = plot_confusion_matrix(clf_sgd, test_x, test_Y, cmap=plt.cm.Greens, values_format='.0f',
                                 xticks_rotation='horizontal')
    plt.title("Confusion Matrix for SGD")

    plt.show()


# AdaBoost Grid Search
def adaboost_grid_search():
    # Creating a grid of different hyperparameters
    grid_params = {
        'n_estimators': [20, 50, 100, 200, 500, 800],
        'learning_rate': [0.05, 0.8, 1]
    }

    # Adaboost classifier
    clf = AdaBoostClassifier(random_state=0)

    print("Searching for optimal parameters..............")

    # Building a 10 fold Cross-Validated GridSearchCV object
    grid_object = GridSearchCV(estimator=clf, param_grid=grid_params, cv=10)

    print("Training the model...............")

    # Fitting the grid to the training data
    grid_object.fit(train_x, train_Y)

    # Extracting the best parameters
    print(grid_object.best_params_)

    # Extracting the best model=
    rf_best = grid_object.best_estimator_
    print(rf_best)


# adaboost
def adaboost_clf():
    print(colored("------Adaboost Classification-------", 'red'))
    # define classifier
    clf_abc = AdaBoostClassifier(n_estimators=15, learning_rate=1)

    # time it
    starttime = timeit.default_timer()

    print("Training the Adaboost Classifier.......")

    # fit data
    clf_abc.fit(train_x, train_Y)

    print("The time difference is :", timeit.default_timer() - starttime)

    print("Predicting test data.......")

    # predict
    pred_y = clf_abc.predict(test_x)

    # results
    c_matrix = confusion_matrix(test_Y, pred_y)
    error = zero_one_loss(test_Y, pred_y)
    score = accuracy_score(test_Y, pred_y)

    # display results
    print('Confusion Matrix\n---------------------------\n', c_matrix)
    print('---------------------------')
    print("Error: {:.4f}%".format(error * 100))
    print("Accuracy Score: {:.4f}%".format(score * 100))
    print(classification_report(test_Y, pred_y))
    print('accuracy: ', c_matrix.diagonal() / c_matrix.sum(axis=1))

    # Plot non-normalized confusion matrix
    disp = plot_confusion_matrix(clf_abc, test_x, test_Y, cmap=plt.cm.Greens, values_format='.0f',
                                 xticks_rotation='horizontal')
    plt.title("Confusion Matrix for AdaBoost")

    plt.show()


# Multi-Layer Percepton MLP
def mlp_clf():
    print(colored("------MLP Classification-------", 'red'))

    # Build classifier
    clf_nn = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(1000, 5), max_iter=1000, random_state=1)

    print("Training the MLP Classifier.......")

    # start timer
    starttime = timeit.default_timer()  # start timer

    # train
    clf_nn.fit(train_x, train_Y)

    print("The time difference is :", timeit.default_timer() - starttime)

    print("Predicting test data.......")

    # predict
    nn_pred = clf_nn.predict(test_x)

    # results
    c_matrix = confusion_matrix(test_Y, nn_pred)
    error = zero_one_loss(test_Y, nn_pred)
    score = accuracy_score(test_Y, nn_pred)

    # display results
    print('Confusion Matrix\n---------------------------\n', c_matrix)
    print('---------------------------')
    print("Error: {:.4f}%".format(error * 100))
    print("Accuracy Score: {:.4f}%".format(score * 100))
    print(classification_report(test_Y, nn_pred))
    print('accuracy: ', c_matrix.diagonal() / c_matrix.sum(axis=1))

    # Plot non-normalized confusion matrix
    disp = plot_confusion_matrix(clf_nn, test_x, test_Y, cmap=plt.cm.Greens, values_format='.0f',
                                 xticks_rotation='horizontal')
    plt.title("Confusion Matrix for Neural Network")

    plt.show()


# Xgboost grid search
def xgboost_grid_search():
    # Creating a grid of different hyperparameters
    grid_params = {"learning_rate": [0.05, 0.10, 0.2, 0.3],
                   "max_depth": [3, 4, 5, 6, 8, 10, 12, 15],
                   "n_estimators": [50, 100, 200, 250, 500]
                   }

    # Adaboost classifier
    clf = xgb.XGBClassifier()

    print("Searching for optimal parameters..............")

    # Building a 10 fold Cross-Validated GridSearchCV object
    grid_object = GridSearchCV(estimator=clf, param_grid=grid_params, cv=5, scoring='accuracy', n_jobs=-1)

    print("Training the model...............")

    # Fitting the grid to the training data
    grid_object.fit(train_x, train_Y)

    # Extracting the best parameters
    print(grid_object.best_params_)

    # Extracting the best model=
    rf_best = grid_object.best_estimator_
    print(rf_best)

    print('Best score : ', grid_object.best_score_)


# xgboost classifier
def xgboost_clf():
    print(colored("------XGBoost Classification-------", 'red'))

    xgb_model = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                                  colsample_bynode=1, colsample_bytree=1, gamma=1,
                                  learning_rate=0.2, max_delta_step=0, max_depth=3,
                                  min_child_weight=1, missing=None, n_estimators=490, n_jobs=-1,
                                  nthread=None, objective='multi:softprob', random_state=0,
                                  reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=0,
                                  silent=None, subsample=1, verbosity=1)

    print("Training the XGBoost Classifier.......")

    # start timer
    starttime = timeit.default_timer()  # start timer

    xgb_model.fit(train_x, train_Y)

    print("The time difference is :", timeit.default_timer() - starttime)

    print("Predicting test data.......")

    # print(xgb_model.feature_importances_)

    xgb_pred = xgb_model.predict(test_x)

    # plot
    # plot_importance(xgb_model, height=0.9)
    # pyplot.show()

    # Feature importance
    '''selector = RFE(xgb_model, 40, step=1)
    selector = selector.fit(train_x, train_Y)
    print(selector.support_)
    print(selector.ranking_)'''

    # results
    c_matrix = confusion_matrix(test_Y, xgb_pred)
    error = zero_one_loss(test_Y, xgb_pred)
    score = accuracy_score(test_Y, xgb_pred)

    # display results
    print('Confusion Matrix\n---------------------------\n', c_matrix)
    print('---------------------------')
    print("Error: {:.4f}%".format(error * 100))
    print("Accuracy Score: {:.4f}%".format(score * 100))
    print(classification_report(test_Y, xgb_pred))
    print('accuracy: ', c_matrix.diagonal() / c_matrix.sum(axis=1))

    # Plot non-normalized confusion matrix
    disp = plot_confusion_matrix(xgb_model, test_x, test_Y, cmap=plt.cm.Greens, values_format='.0f',
                                 xticks_rotation='horizontal')
    plt.title("Confusion Matrix for XGBoost")

    plt.show()


# voting classifier
def votingClassifier():
    print(colored("------Voting Classification-------", 'red'))

    # models
    random_forest = RandomForestClassifier(criterion='entropy', max_depth=30, n_estimators=48, random_state=0)
    clf_lr = LogisticRegression()
    clf_knn = KNeighborsClassifier(n_neighbors=7)
    # build classifier
    model = VotingClassifier(estimators=[('rf', random_forest), ('knn', clf_knn)], voting='soft',
                             n_jobs=-1, weights=[2, 1])

    print("Training the Voting classification.......")

    # start timer
    starttime = timeit.default_timer()  # start timer

    cnn = CondensedNearestNeighbour(random_state=42)  # doctest: +SKIP

    # train
    model.fit(train_x, train_Y)

    print("The time difference is :", timeit.default_timer() - starttime)

    print("Predicting test data.......")

    # predict
    y_pred = model.predict(test_x)

    # results
    c_matrix = confusion_matrix(test_Y, y_pred)
    error = zero_one_loss(test_Y, y_pred)
    score = accuracy_score(test_Y, y_pred)

    # display results
    print('Confusion Matrix\n---------------------------\n', c_matrix)
    print('---------------------------')
    print("Error: {:.4f}%".format(error * 100))
    print("Accuracy Score: {:.4f}%".format(score * 100))
    print(classification_report(test_Y, y_pred))
    print('accuracy: ', c_matrix.diagonal() / c_matrix.sum(axis=1))

    # Plot non-normalized confusion matrix
    disp = plot_confusion_matrix(model, test_x, test_Y, cmap=plt.cm.Greens, values_format='.0f',
                                 xticks_rotation='horizontal')
    plt.title("Confusion Matrix for Voting Classifier")

    plt.show()


# lightgbm

def lightgbm_clf():
    print(colored("------Lightgbm Classification-------", 'red'))

    # model
    model = lgb.LGBMClassifier(objective='binary', boosting_type='goss', n_estimators=10000, class_weight='balanced')

    print("Training the Lightgbm classification.......")

    # start timer
    starttime = timeit.default_timer()  # start timer

    model.fit(train_x, train_Y)

    print("The time difference is :", timeit.default_timer() - starttime)

    print("Predicting test data.......")

    # predict
    y_pred = model.predict(test_x)

    # results
    c_matrix = confusion_matrix(test_Y, y_pred)
    error = zero_one_loss(test_Y, y_pred)
    score = accuracy_score(test_Y, y_pred)

    # display results
    print('Confusion Matrix\n---------------------------\n', c_matrix)
    print('---------------------------')
    print("Error: {:.4f}%".format(error * 100))
    print("Accuracy Score: {:.4f}%".format(score * 100))
    print(classification_report(test_Y, y_pred))
    print('accuracy: ', c_matrix.diagonal() / c_matrix.sum(axis=1))

    # Plot non-normalized confusion matrix
    disp = plot_confusion_matrix(model, test_x, test_Y, cmap=plt.cm.Greens, values_format='.0f',
                                 xticks_rotation='horizontal')

    plt.title("Confusion Matrix for Lightgbm")

    plt.show()


def catboost_clf():
    print(colored("------Catboost Classification-------", 'red'))

    # drop less important features
    x_train.drop(
        columns=['urgent', 'root_shell', 'num_shells', 'dst_host_srv_rerror_rate', 'su_attempted',
                 'srv_serror_rate', 'num_access_files', 'srv_diff_host_rate', 'is_host_login', 'logged_in',
                 'srv_rerror_rate'], axis=1, inplace=True)

    x_test.drop(
        columns=['urgent', 'root_shell', 'num_shells', 'dst_host_srv_rerror_rate', 'su_attempted',
                 'srv_serror_rate', 'num_access_files', 'srv_diff_host_rate', 'is_host_login', 'logged_in',
                 'srv_rerror_rate'], axis=1, inplace=True)

    # model
    model = cb.CatBoostClassifier(iterations=490, cat_features=nominal_cols, learning_rate=0.3, l2_leaf_reg=1,
                                  max_depth=2, bootstrap_type='Bayesian', bagging_temperature=1)

    print("Training the Catboost classification.......")

    # start timer
    starttime = timeit.default_timer()  # start timer

    # train
    model.fit(x_train, train_Y)

    print("The time difference is :", timeit.default_timer() - starttime)

    print("Predicting test data.......")

    feature_importances = model.get_feature_importance()
    feature_names = x_train.columns
    for score, name in sorted(zip(feature_importances, feature_names), reverse=True):
        print('{}: {}'.format(name, score))

    # predict
    y_pred = model.predict(x_test)

    # results
    c_matrix = confusion_matrix(test_Y, y_pred)
    error = zero_one_loss(test_Y, y_pred)
    score = accuracy_score(test_Y, y_pred)

    # display results
    print('Confusion Matrix\n---------------------------\n', c_matrix)
    print('---------------------------')
    print("Error: {:.4f}%".format(error * 100))
    print("Accuracy Score: {:.4f}%".format(score * 100))
    print(classification_report(test_Y, y_pred))
    print('accuracy: ', c_matrix.diagonal() / c_matrix.sum(axis=1))

    # Plot non-normalized confusion matrix
    disp = plot_confusion_matrix(model, x_test, test_Y, cmap=plt.cm.Greens, values_format='.0f',
                                 xticks_rotation='horizontal')

    plt.title("Confusion Matrix for Catboost")

    plt.show()


if __name__ == "__main__":
    # grid search function -- takes long time to run
    # random_forest_grid_search()
    # knn_grid_search()
    # find_k()
    # svc_grid_search()
    # logistic_reg_grid_search()
    # sgd_grid_search()
    # adaboost_grid_search()
    # xgboost_grid_search()
    # classifiers
    decision_tree_clf()
    data_exploration_solution()
    random_forest_clf()
    knn_clf()
    svm_clf()
    logistic_reg_clf()
    sgd_clf()
    adaboost_clf()
    mlp_clf()
    xgboost_clf()
    votingClassifier()
    lightgbm_clf()
    catboost_clf()
